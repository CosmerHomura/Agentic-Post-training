"""
第 1 步 · 监督微调（SFT）—— 全流程可视化版

一条轨迹进模型之前，要变四次形：

    JSONL 记录  --chat template-->  一段纯文本  --tokenizer-->  token id 序列
        --掩码-->  labels（要学的位置保留 id，其余 -100）  --collate-->  一个 batch 张量

本脚本把每一步都打印出来，最后跑训练、画损失、做训练前后对比。

用法
----
    python train_sft.py --inspect      # 只跑数据链路（不需要 GPU，甚至不需要 trl/torch）
    python train_sft.py --dataset trl-lib/Capybara --limit 2000   # 用官方数据集真训
    python train_sft.py                # 用本地 2 条解剖样本跑一遍全流程
    python train_sft.py --epochs 1 --model Qwen/Qwen3-1.7B --max-steps 200

本地那 2 条轨迹只用来「看清数据长什么样」；要看训练效果，就得用官方数据集
（`trl-lib/Capybara`，对话格式，1.6 万条）或者你自己的 agent 轨迹。

评测：前 --eval-size 条留出来不训练，训练前后各算一次 **held-out 上的平均 -ln p**
（越小说明模型越能预测人类答案），再跑几个生成样例做肉眼对照。

产物
----
    runs/sft/report.md     给人看的总报告（把这份发给我，我来更新教程里的示例输出）
    runs/sft/report.json   同样的内容，机器可读
    runs/sft/tokens.txt    全量 token/label/mask 表（终端只打印开头的窗口）
    runs/sft/loss.csv      每步 loss，可拿去画图
    ckpt/sft-agent/        训练好的权重

数据
----
    ../data/sft_agent_trace.jsonl   每行 {"tools": [...], "messages": [...]}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "sft_agent_trace.jsonl"
RUN_DIR = HERE.parent / "runs" / "sft"

DEFAULT_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")


# ============================================================ 0. 小工具
def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def kv(**kw) -> None:
    for k, v in kw.items():
        print(f"  {k:<22} {v}")


def piece(tok, tid: int) -> str:
    """
    把 token 还原成可读文本。

    注意：Qwen 用的是 byte-level BPE，convert_ids_to_tokens 给的是「字节映射后的伪 latin 串」，
    直接把那个字符串打出来会是乱码；用 decode 才是人类可读的文本。
    单个 token 可能只是某个汉字的一个字节，decode 出来就是 � —— 这不是 bug，正是 tokenizer 的现实。
    """
    s = tok.decode([tid], skip_special_tokens=False)
    return s.replace(" ", "␣").replace("\n", "\\n").replace("\t", "\\t")


def ids_of(tok, messages, tools, **kw):
    """apply_chat_template 在 transformers 4.x 返回 list、5.x 返回 dict，这里统一成 list。"""
    out = tok.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=False, **kw
    )
    return list(out["input_ids"]) if hasattr(out, "keys") else list(out)


def patch_template(tpl: str) -> tuple[str, bool]:
    """
    给 chat template 插入 {% generation %} / {% endgeneration %}。

    为什么需要：transformers 只有看到这对标记，才知道「哪几个 token 是模型该生成的」，
    才会返回 assistant_masks。Qwen3 的模板里没有这对标记，所以直接调
    return_assistant_tokens_mask=True 会得到全 0（并给一条 warning）。
    TRL 的 assistant_only_loss=True 在内部做的也是同一件事：打补丁 → 取掩码。
    """
    if "{% generation %}" in tpl or "{%- generation %}" in tpl:
        return tpl, False
    patched = tpl.replace(
        '{%- elif message.role == "assistant" %}',
        '{%- elif message.role == "assistant" %}{%- generation %}',
    )
    patched = patched.replace(
        '{%- elif message.role == "tool" %}',
        '{%- endgeneration %}{%- elif message.role == "tool" %}',
    )
    if "{%- generation %}" not in patched:
        raise RuntimeError("模板结构不认识，没法自动打 generation 补丁；请手动检查 chat_template。")
    return patched, True


def build_masks(tok, messages, tools):
    """
    返回 (input_ids, assistant_mask, 说明)。

    规则（与 TRL 的 assistant_only_loss 同源）：
      - assistant 轮：监督（含它的 <tool_call> JSON 和结尾的 <|im_end|>）
      - user / system / tool 轮：不监督
    另外把每个 assistant 轮开头的 <|im_start|>assistant\n 三个 token 排除，
    因为那是「给定上下文」，不是模型要说的话。
    """
    tpl, was_patched = patch_template(tok.chat_template)
    old = tok.chat_template
    tok.chat_template = tpl
    try:
        out = tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        ids = list(out["input_ids"])
        mask = list(out["assistant_masks"])
    finally:
        tok.chat_template = old

    if sum(mask) == 0:
        raise RuntimeError("掩码全 0：模板补丁没生效，检查 chat_template 结构。")

    # 去掉每段连续「1」最前面的 3 个 token（<|im_start|> / assistant / 换行）
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            for k in range(i, min(i + 3, j)):
                mask[k] = 0
            i = j
        else:
            i += 1

    note = "模板自带 generation 标记" if not was_patched else "已给模板打 generation 补丁"
    return ids, mask, note


# ============================================================ 1. 读数据
def load_records(args):
    """
    返回 (训练集, 评测集, 来源说明)。

    --dataset 给 HF 数据集名时走官方数据（Capybara 的列叫 messages，和本地 jsonl 同构）；
    否则读本地 jsonl —— 那 2 条是「解剖样本」，用来看模板和掩码，不足以训练出能力。
    """
    if args.dataset:
        from datasets import load_dataset

        print(f"  从 HF 加载：{args.dataset}（split={args.split}）")
        try:
            ds = load_dataset(args.dataset, split=args.split)
        except Exception as e:  # 网络/鉴权/名字错，给一句人话
            raise SystemExit(
                f"  加载失败：{e}\n"
                f"  国内网络先设 HF_ENDPOINT=https://hf-mirror.com 再跑；"
                f"或者用 --data 指向本地 jsonl。"
            ) from e
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        recs = [{"messages": r["messages"], "tools": r.get("tools")} for r in ds]
        src = f"{args.dataset}（取前 {len(recs)} 条）"
    else:
        path = Path(args.data)
        recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        src = str(path)

    n_eval = min(args.eval_size, max(1, len(recs) // 10)) if len(recs) >= 4 else 0
    return recs[n_eval:], recs[:n_eval], src


def stage_read(path, records, eval_records):
    hr("阶段 1 / 4   读数据")
    print(f"  来源：{path}")
    print(f"  训练：{len(records)} 条    评测（held-out）：{len(eval_records)} 条")
    print()
    print(f"  {'#':>2}  {'role':<10} {'content 长度':>12}  tool_calls")
    print("  " + "-" * 62)
    for i, m in enumerate(records[0]["messages"]):
        tc = m.get("tool_calls") or []
        names = ",".join(t["function"]["name"] for t in tc) if tc else "-"
        print(f"  {i:>2}  {m['role']:<10} {len(m.get('content') or ''):>12}  {names}")
    print()
    print("  第 1 条记录的 tools 列（渲染进 system 消息里的函数签名）：")
    for t in records[0].get("tools", []):
        f = t["function"]
        print(f"    - {f['name']}({', '.join(f['parameters'].get('properties', {}))})  # {f['description']}")
    if not records[0].get("tools"):
        print("    （这条没有工具定义 —— 官方 Capybara 就是纯对话，想要 agent 轨迹得自己采集）")
    return records


def answer_span(rec):
    """最后一条 assistant 的 content 长度，用来在评测里判断「模型有没有学会那种格式」。"""
    for m in reversed(rec["messages"]):
        if m["role"] == "assistant":
            return len(m.get("content") or "")
    return 0



# ============================================================ 2. 套模板
def stage_template(tok, rec):
    hr("阶段 2 / 4   套 chat template（JSON 结构 → 一段纯文本）")
    msgs, tools = rec["messages"], rec.get("tools")
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
    print(f"  渲染结果：{len(text)} 个字符")
    print("  下面原样打印；每行开头的 <|im_start|>role 是模板插的分隔符：")
    print("  " + "-" * 74)
    for line in text.splitlines():
        mark = "  "
        if line.startswith("<|im_start|>"):
            mark = "▸ "
        elif line.startswith("<|im_end|>"):
            mark = "◂ "
        print(f"  {mark}{line}")
    print("  " + "-" * 74)
    return text


# ============================================================ 3. token / label / mask
def stage_tokens(tok, rec, show_window=14):
    hr("阶段 3 / 4   tokenize → 打掩码 → labels")
    msgs, tools = rec["messages"], rec.get("tools")
    ids, mask, note = build_masks(tok, msgs, tools)
    labels = [t if m else -100 for t, m in zip(ids, mask)]

    n_sup = sum(mask)
    print(f"  {note}")
    print(f"  总 token：{len(ids)}    需要学的 token：{n_sup}（{n_sup / len(ids):.1%}）")
    print()

    # 掩码条：一个字符一个 token
    print("  掩码条（█ = 要学，· = 不学，每行 100 个 token）：")
    for base in range(0, len(mask), 100):
        chunk = mask[base : base + 100]
        print(f"    {base:>4} " + "".join("█" if m else "·" for m in chunk))
    print()

    # 表格：开头一段 + 第一处「开始学」的位置附近 + 结尾一段
    head = list(range(min(10, len(ids))))
    first = mask.index(1) if 1 in mask else 0
    mid = list(range(max(0, first - 6), min(len(ids), first + 16)))
    tail = list(range(max(0, len(ids) - 10), len(ids)))
    shown = sorted(set(head + mid + tail))
    print(f"  {'idx':>4}  {'token_id':>8}  {'label':>8}  {'学':<2}  token")
    print("  " + "-" * 58)
    prev = None
    for i in shown:
        if prev is not None and i != prev + 1:
            print(f"  {'…':>4}  {'…':>8}  {'…':>8}     …")
        flag = "█" if mask[i] else "·"
        print(f"  {i:>4}  {ids[i]:>8}  {labels[i]:>8}  {flag:<2}  {piece(tok, ids[i])}")
        prev = i
    print()
    print("  上面这一段就是「同一个 token 在第 1 行是原文、在第 3 行变成 -100」的地方：")
    print("  掩码翻转的位置，正好落在 <|im_start|>assistant 之后。")

    full = RUN_DIR / "tokens.txt"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(
        "\n".join(
            f"{i:>5}  {t:>8}  {l:>8}  {'1' if m else '0'}  {piece(tok, t)}"
            for i, (t, l, m) in enumerate(zip(ids, labels, mask))
        ),
        encoding="utf-8",
    )
    print(f"\n  全量表已写入：{full}")
    return ids, mask, labels


# ============================================================ 4. 一个 batch
def stage_batch(tok, records):
    hr("阶段 4 / 4   拼成 batch（模型真正看到的东西）")
    batch_ids, batch_lab, batch_am = [], [], []
    for rec in records:
        ids, mask, _ = build_masks(tok, rec["messages"], rec.get("tools"))
        batch_ids.append(ids)
        batch_lab.append([t if m else -100 for t, m in zip(ids, mask)])
        batch_am.append([1] * len(ids))

    maxlen = max(len(x) for x in batch_ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for ids, lab, am in zip(batch_ids, batch_lab, batch_am):
        pad_n = maxlen - len(ids)
        ids += [pad] * pad_n
        lab += [-100] * pad_n     # padding 位置必须 -100，否则会去学 pad
        am += [0] * pad_n

    print(f"  样本数：{len(batch_ids)}   序列长度（pad 后）：{maxlen}")
    print()
    shape = f"[{len(batch_ids)}, {maxlen}]"
    kv(
        input_ids=f"{shape}  int64   # pad={pad}",
        attention_mask=f"{shape}  int64   # pad 处为 0",
        labels=f"{shape}  int64   # 不学的位置 = -100",
    )
    print()
    print("  第 1 条样本的头 12 个位置：")
    print(f"  {'idx':>4}  {'input_ids':>10}  {'labels':>10}  {'attn':>4}")
    for i in range(12):
        print(f"  {i:>4}  {batch_ids[0][i]:>10}  {batch_lab[0][i]:>10}  {batch_am[0][i]:>4}")
    print()
    print("  注意：labels 只是把不该学的位置换成 -100（CrossEntropy 的 ignore_index），")
    print("        其余位置和 input_ids 完全相同 —— 这就是「右移一位」的自回归目标。")
    try:
        import torch

        t = torch.tensor(batch_ids, dtype=torch.long)
        print(f"\n  torch 版：{tuple(t.shape)}  {t.dtype}  设备无关，训练时由 Trainer 搬到 GPU")
    except ImportError:
        print("\n  （没装 torch，上面用纯 Python 列表演示形状；训练阶段需要 torch）")


# ============================================================ 5. 训练
def train(model_name, records, eval_records, epochs, max_steps, bf16, before_hook=None):
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
        def __init__(self):
            self.rows = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            self.rows.append(
                {
                    "step": state.global_step,
                    "loss": round(logs["loss"], 4),
                    "lr": float(logs.get("learning_rate", 0.0)),
                    "grad_norm": round(float(logs.get("grad_norm", 0.0) or 0.0), 4),
                    "tok_acc": round(float(logs.get("mean_token_accuracy", 0.0) or 0.0), 4),
                }
            )
            r = self.rows[-1]
            print(f"  step {r['step']:>3}  loss {r['loss']:.4f}  lr {r['lr']:.2e}  "
                  f"grad {r['grad_norm']:.3f}  token_acc {r['tok_acc']:.3f}")

    printer = LossPrinter()
    args = SFTConfig(
        output_dir=str(HERE.parent / "ckpt" / "sft-agent"),
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
        report_to=[],
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

    # ---- 训练前：先在 held-out 上量一次，并抓生成样例 ----
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


def chart(rows, width=46):
    """把 loss 序列画成 ASCII 走势图（终端里也能看见「在下降」）。"""
    losses = [r["loss"] for r in rows]
    if len(losses) < 2:
        return ""
    lo, hi = min(losses), max(losses)
    span = (hi - lo) or 1.0
    out = [f"  loss 走势 (min {lo:.3f} / max {hi:.3f})"]
    idx = [round(i * (len(losses) - 1) / (width - 1)) for i in range(width)]
    for level in range(6, -1, -1):
        thr = lo + span * level / 6
        line = "".join("●" if losses[i] >= thr else " " for i in idx)
        out.append(f"  {thr:6.3f} |{line}")
    out.append("         +" + "-" * width)
    out.append(f"          step {rows[0]['step']} → {rows[-1]['step']}")
    return "\n".join(out)


# ============================================================ 6. 训练前后对比
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
        xpected = (msgs[cut].get("content") or "") if cut < len(msgs) else ""
        cases.append((prompt, rec.get("tools"), xpected))
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
        print(f"\n  {tag}：格式合规 {ok}/{len(outs)}（{'tool_call JSON 可解析' if outs[0] and tools else '非空输出'}）")
    return outs


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


# ============================================================ 7. 报告
def write_report(meta, before, after, rows):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# SFT 运行报告", ""]
    for k, v in meta.items():
        md.append(f"- **{k}**：{v}")
    md += ["", "## 损失", "", "| step | loss | lr | grad_norm | token_acc |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['step']} | {r['loss']:.4f} | {r['lr']:.2e} | {r['grad_norm']:.3f} | {r['tok_acc']:.3f} |")
    if rows:
        md += ["", "```", chart(rows), "```"]
    md += ["", "## 训练前 vs 训练后（held-out 生成）", ""]
    for b, a in zip(before, after):
        md += [f"### prompt：`{b['prompt'][:120]}`", "", "**训练前**：", "```", b["output"], "```", ""]
        md += ["**训练后**：", "```", a["output"], "```", ""]
        md += [f"- 参考（数据里的真实答案）：`{b['expected']}`", f"- 训练后格式合规：{a['format_ok']}", ""]
    (RUN_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")
    (RUN_DIR / "report.json").write_text(
        json.dumps({"meta": meta, "loss": rows, "before": before, "after": after}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RUN_DIR / "loss.csv").write_text(
        "step,loss,lr,grad_norm,token_acc\n"
        + "\n".join(f"{r['step']},{r['loss']},{r['lr']},{r['grad_norm']},{r['tok_acc']}" for r in rows),
        encoding="utf-8",
    )
    print(f"\n  报告：{RUN_DIR / 'report.md'}")
    print("  把 report.md 发给我，我来把真实输出补进教程。")


# ============================================================ 手算损失
def explain_loss() -> None:
    """不加载模型，把 token 级交叉熵一项一项算出来。"""
    import math

    hr("手算一遍：SFT 的损失到底是什么（交叉熵）")
    print("  模型在每个位置输出一整个词表的分数（logits），softmax 之后是这个 token 的概率。")
    print("  损失 = 正确 token 的 -ln(概率)。猜得越准，损失越小。")
    print()
    print(f"  {'正确 token 的概率':>16}  {'-ln(p)':>8}   直觉")
    print("  " + "-" * 56)
    for p, note in [
        (0.99, "几乎确定 → 几乎不罚"),
        (0.90, "很确定"),
        (0.50, "在犹豫"),
        (0.20, "猜错了"),
        (0.01, "完全跑偏 → 罚得很重"),
    ]:
        print(f"  {p:>16.2f}  {-math.log(p):>8.3f}   {note}")
    print()
    print("  一条轨迹里 5 个被监督的 token，假设模型给它们这些概率：")
    print()
    ps = [0.98, 0.95, 0.90, 0.80, 0.50]
    for i, p in enumerate(ps):
        print(f"    token {i + 1}: p = {p:.2f}   →   -ln(p) = {-math.log(p):.3f}")
    mean = sum(-math.log(p) for p in ps) / len(ps)
    print(f"    平均 = {mean:.3f}   ← 这就是 loss（这个 batch 里所有被监督 token 的平均）")
    print()
    print("  关键三点：")
    print("   1) 只有 label != -100 的位置参与 —— prompt 和 tool 结果不算（第 5 节的掩码条）。")
    print("   2) 位置 t 的预测要和位置 t+1 的 token 比（右移一位），不是和它自己比。")
    print("   3) 损失对所有被监督 token 取平均，所以长轨迹里的短片段会被稀释。")


# ============================================================ main
def main():
    global _TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="只跑数据链路，不训练")
    ap.add_argument("--explain-loss", action="store_true", help="玩具数字手算交叉熵（不需 GPU）")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--dataset", default=None, help="HF 数据集名，如 trl-lib/Capybara（优先于 --data）")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=2000, help="从官方数据集取前多少条")
    ap.add_argument("--eval-size", type=int, default=100, help="留出多少条做 held-out 评测")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true", help="跳过训练前后生成对比")
    ap.add_argument("--gen-tokens", type=int, default=220, help="生成对比时最多新生成多少 token")
    ap.add_argument("--fp32", action="store_true", help="不用 bf16")
    args = ap.parse_args()

    if args.explain_loss:
        explain_loss()
        return

    hr("阶段 0 / 4   环境")
    import transformers

    print(f"  python        {sys.version.split()[0]}")
    print(f"  transformers  {transformers.__version__}")
    try:
        import trl

        print(f"  trl           {trl.__version__}")
    except ImportError:
        print("  trl           (未安装 —— 只能跑 --inspect)")
    try:
        import torch

        print(f"  torch         {torch.__version__}   cuda={torch.cuda.is_available()}")
    except ImportError:
        print("  torch         (未安装)")
    print(f"  数据          {args.dataset or args.data}")

    records, eval_records, src = load_records(args)
    stage_read(src, records, eval_records)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"\n  tokenizer     {type(tok).__name__}  vocab={len(tok)}  "
          f"chat_template={len(tok.chat_template or '')} 字符")

    # 解剖：始终用第一条训练样本，保证「模板/token/掩码」讲的是同一份数据
    stage_template(tok, records[0])
    stage_tokens(tok, records[0])
    stage_batch(tok, records[:2])

    if args.inspect:
        print("\n  --inspect 模式结束（没有训练）。去掉 --inspect 就跑完整流程。")
        return

    def grab_samples(model):
        model.eval()
        return stage_before_after(tok, model, eval_records, "生成", args.gen_tokens)

    trainer, rows, pre, post, before, after = train(
        args.model,
        records,
        [] if args.no_eval else eval_records,
        args.epochs,
        args.max_steps,
        bf16=not args.fp32,
        before_hook=None if args.no_eval else grab_samples,
    )
    if rows:
        print("\n" + chart(rows))
    if pre and post:
        d = pre["eval_loss"] - post["eval_loss"]
        print(f"\n  held-out -ln p：{pre['eval_loss']:.4f} → {post['eval_loss']:.4f}"
              f"（{'下降' if d > 0 else '上升'} {abs(d):.4f}）")

    meta = {
        "模型": args.model,
        "数据": f"{src}（训练 {len(records)} 条 / 评测 {len(eval_records)} 条）",
        "epochs": args.epochs,
        "held-out -ln p（训练前）": f"{pre['eval_loss']:.4f}" if pre else "未评测",
        "held-out -ln p（训练后）": f"{post['eval_loss']:.4f}" if post else "未评测",
        "耗时(s)": round(trainer.state.log_history[-1].get("train_runtime", 0), 1)
        if trainer.state.log_history
        else "n/a",
    }
    write_report(meta, before, after, rows)


if __name__ == "__main__":
    main()
