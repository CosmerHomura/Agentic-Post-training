"""
训练 + 评测 + 生成对比。

拆出来的理由：数据/模板在别的文件里已经讲清楚了，这里只说「模型怎么被更新、
怎么判断它真的学会了」——训练前的 held-out 指标必须在 train() 之前取，
这是之前写错过一次的地方（先训练再量「训练前」，数字毫无意义）。
"""

from __future__ import annotations

import json
import re
import sys
import time

from common import CKPT_DIR, RUN_DIR, fmt_eta, hr, plot_svg, spark


def eval_sets(eval_records, k=2):
    """
    拿 held-out 里的真实样本做生成对比：把最后一条 assistant 去掉当 prompt，
    让模型自己写那一段 —— 它写出来的东西才是「学会了没有」的直接证据。
    """
    cases = []
    for rec in eval_records[:k]:
        msgs = rec["messages"]
        cut = len(msgs) - 1
        while cut > 0 and msgs[cut - 1]["role"] != "user":
            cut -= 1
        prompt = msgs[:cut] if cut > 0 else msgs[:1]
        expected = (msgs[cut].get("content") or "") if cut < len(msgs) else ""
        cases.append((prompt, rec.get("tools"), expected))
    return cases


def toolcall_json_ok(text: str) -> bool:
    """输出里有没有一个能 json.loads 的 <tool_call>{...}</tool_call> —— agent 场景的格式分。"""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:
        return False
    try:
        json.loads(m.group(1))
        return True
    except Exception:
        return False


def generate(tok, model, messages, tools, max_new_tokens=220):
    import torch

    prompt = tok.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    input_ids = prompt["input_ids"]
    input_ids = torch.tensor([input_ids], device=model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)


def stage_before_after(tok, model, eval_records, tag, max_new_tokens=220):
    hr(f"阶段 6   生成对比（{tag}，held-out 样本）")
    outs, ok = [], 0
    for messages, tools, expected in eval_sets(eval_records):
        text = generate(tok, model, messages, tools, max_new_tokens)
        hit = toolcall_json_ok(text) if tools else bool(text.strip())
        ok += int(hit)
        print(f"\n  ── prompt：{(messages[-1].get('content') or '')[:60]} ──")
        print("  模型输出：")
        for line in text.strip().splitlines()[:12]:
            print(f"    {line}")
        outs.append(
            {"prompt": messages[-1].get("content", ""), "output": text.strip(),
             "expected": expected[:200], "format_ok": hit}
        )
    if outs:
        label = "tool_call JSON 可解析" if tools else "非空输出"
        print(f"\n  {tag}：格式合规 {ok}/{len(outs)}（{label}）")
    return outs


def train(model_name, records, eval_records, epochs, max_steps, bf16, plot_every=20,
          before_hook=None, tb=False):
    hr("阶段 5   训练（每个 logging 步打印 loss）")
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_list([{"messages": r["messages"], "tools": r.get("tools")} for r in records])
    eval_ds = (
        Dataset.from_list([{"messages": r["messages"], "tools": r.get("tools")} for r in eval_records])
        if eval_records
        else None
    )

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        # CPU 上 bf16 不被支持（transformers 会直接报 use_cpu），顺便换掉 fused 优化器
        bf16 = False
    print(f"  设备：{'GPU' if has_cuda else 'CPU（小样本冒烟用，会很慢）'}   bf16：{bf16}")

    class LossPrinter(TrainerCallback):
        """每步一行实时进度；每 plot_every 步把曲线重写成 runs/sft/loss.svg。"""

        def __init__(self):
            self.rows = []
            self.ema = None
            self.t0 = time.time()
            self.tty = sys.stdout.isatty()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            r = {
                "step": state.global_step,
                "loss": round(float(logs["loss"]), 4),
                "lr": float(logs.get("learning_rate", 0.0)),
                "grad_norm": round(float(logs.get("grad_norm", 0) or 0), 4),
                "tok_acc": round(float(logs.get("mean_token_accuracy", 0) or 0), 4),
            }
            self.rows.append(r)
            self.ema = r["loss"] if self.ema is None else 0.9 * self.ema + 0.1 * r["loss"]
            total = state.max_steps or 0
            pct = 100.0 * r["step"] / total if total else 0.0
            eta = ""
            if total and r["step"]:
                eta = "  eta " + fmt_eta((time.time() - self.t0) / r["step"] * (total - r["step"]))
            line = (f"step {r['step']:>4}/{total:<4} {pct:5.1f}%  loss {r['loss']:.4f}  "
                    f"ema {self.ema:.4f}  acc {r['tok_acc']:.3f}  lr {r['lr']:.2e}{eta}  "
                    f"{spark([x['loss'] for x in self.rows])}")
            if self.tty:
                print("\r\033[K" + line[:160], end="", flush=True)
            else:
                print("  " + line, flush=True)
            if plot_every and r["step"] % plot_every == 0:
                self._plot(total)

        def _plot(self, total):
            steps = [x["step"] for x in self.rows]
            losses = [x["loss"] for x in self.rows]
            ema, e = [], None
            for v in losses:
                e = v if e is None else 0.9 * e + 0.1 * v
                ema.append(e)
            plot_svg(
                RUN_DIR / "loss.svg",
                [
                    {"name": "每步 loss", "pts": list(zip(steps, losses)), "color": "#93c5fd", "width": 1.2},
                    {"name": "EMA(0.9)", "pts": list(zip(steps, ema)), "color": "#2563eb", "width": 2.0},
                ],
                title=f"loss 曲线（{steps[-1]}/{total or steps[-1]} 步）", ylabel="loss",
            )

        def on_train_end(self, args, state, control, **kwargs):
            if self.tty:
                print()
            if self.rows:
                self._plot(state.max_steps or self.rows[-1]["step"])

    printer = LossPrinter()
    args = SFTConfig(
        output_dir=str(CKPT_DIR),
        model_init_kwargs={"dtype": "bfloat16" if bf16 else "float32"},
        use_cpu=not has_cuda,
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",
        assistant_only_loss=True,   # 只对 assistant 轮算 loss（和阶段 3 的掩码一致）
        packing=False,              # 先不开 packing，保证「一条轨迹 = 一条序列」，便于对照
        max_length=4096,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_steps=3,             # TRL 1.16 的 SFTConfig 只有 warmup_steps（没有 warmup_ratio）
        lr_scheduler_type="cosine",
        bf16=bf16,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to=["tensorboard"] if tb else [],
        logging_dir=str(RUN_DIR / "tb"),
        run_name=RUN_DIR.name,
    )
    trainer = SFTTrainer(
        model=model_name,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        callbacks=[printer],
    )

    n_param = sum(p.numel() for p in trainer.model.parameters())
    print(f"  模型：{model_name}   参数量：{n_param / 1e6:.0f}M")
    print(f"  device：{trainer.model.device}   bf16：{bf16}")
    print()

    # ---- 训练前：先在 held-out 上量一次，并抓生成样例（必须在 train 之前）----
    pre = {}
    if eval_ds is not None:
        pre = trainer.evaluate()
        print(f"  训练前 held-out：loss {pre['eval_loss']:.4f}  （平均 -ln p，越小越好）")
    before = before_hook(trainer.model) if before_hook else []

    print()
    result = trainer.train()
    trainer.save_model(args.output_dir)

    # ---- 训练后 ----
    post = trainer.evaluate() if eval_ds is not None else {}
    if post:
        print(f"  训练后 held-out：loss {post['eval_loss']:.4f}")
    after = before_hook(trainer.model) if before_hook else []

    print()
    print(f"  训练结束：{result.metrics}")
    print(f"  权重已存：{args.output_dir}")
    return trainer, printer.rows, pre, post, before, after
