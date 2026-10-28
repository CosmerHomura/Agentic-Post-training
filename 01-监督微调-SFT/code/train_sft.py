"""
第 1 步 · 监督微调（SFT）—— 全流程可视化版

一条轨迹进模型之前，要变四次形：

    JSONL 记录  --chat template-->  一段纯文本  --tokenizer-->  token id 序列
        --掩码-->  labels（要学的位置保留 id，其余 -100）  --collate-->  一个 batch 张量

本脚本把每一步都打印出来，最后跑训练、画损失、做训练前后对比。

用法
----
    python train_sft.py --inspect      # 只跑数据链路（不需要 GPU，甚至不需要 trl/torch）
    python train_sft.py                # 数据链路 + 训练 + 训练前后对比 + 写报告
    python train_sft.py --epochs 1 --model Qwen/Qwen3-0.6B --max-steps 20

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
def stage_read(path: Path):
    hr("阶段 1 / 4   读 JSONL")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"  文件：{path}")
    print(f"  行数：{len(records)}")
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
    return records


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
def train(model_name: str, records, epochs: float, max_steps: int | None, bf16: bool):
    hr("阶段 5   训练（每个 logging 步打印 loss）")
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_list([{"messages": r["messages"], "tools": r.get("tools")} for r in records])

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
        callbacks=[printer],
    )

    n_param = sum(p.numel() for p in trainer.model.parameters())
    print(f"  模型：{model_name}   参数量：{n_param / 1e6:.0f}M")
    print(f"  device：{trainer.model.device}   bf16：{bf16}")
    print()

    result = trainer.train()
    trainer.save_model(args.output_dir)

    print()
    print(f"  训练结束：{result.metrics}")
    print(f"  权重已存：{args.output_dir}")
    return trainer, printer.rows


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
EVAL_CASES = [
    ("同一类任务（修 bug）", 0),
    ("同格式新任务（补乘除）", 1),
]


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


def eval_sets(records):
    """把轨迹里的「前半段」当作 prompt：system + user（+ 也许还有一轮工具结果）。"""
    cases = []
    for name, idx in EVAL_CASES:
        rec = records[idx]
        msgs = [m for m in rec["messages"] if m["role"] in ("system", "user")][:2]
        cases.append((name, msgs, rec.get("tools")))
    return cases


def stage_before_after(tok, model, records, tag, max_new_tokens=220):
    hr(f"阶段 6   生成对比（{tag}）")
    outs = []
    for name, messages, tools in eval_sets(records):
        text = generate(tok, model, messages, tools, max_new_tokens)
        print(f"\n  ── {name} ──")
        print(f"  prompt（最后一句）：{messages[-1]['content'][:60]}")
        print("  模型输出：")
        for line in text.strip().splitlines()[:12]:
            print(f"    {line}")
        outs.append({"case": name, "prompt": messages[-1]["content"], "output": text.strip()})
    return outs


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
    md += ["", "## 训练前 vs 训练后", ""]
    for b, a in zip(before, after):
        md += [f"### {b['case']}", "", f"**prompt**：`{b['prompt']}`", "", "**训练前**：", "```", b["output"], "```", ""]
        md += ["**训练后**：", "```", a["output"], "```", ""]
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


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="只跑数据链路，不训练")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true", help="跳过训练前后生成对比")
    ap.add_argument("--gen-tokens", type=int, default=220, help="生成对比时最多新生成多少 token")
    ap.add_argument("--fp32", action="store_true", help="不用 bf16")
    args = ap.parse_args()

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
    print(f"  数据          {args.data}")

    records = stage_read(Path(args.data))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"\n  tokenizer     {type(tok).__name__}  vocab={len(tok)}  "
          f"chat_template={len(tok.chat_template or '')} 字符")

    stage_template(tok, records[0])
    stage_tokens(tok, records[0])
    stage_batch(tok, records)

    if args.inspect:
        print("\n  --inspect 模式结束（没有训练）。去掉 --inspect 就跑完整流程。")
        return

    trainer, rows = train(args.model, records, args.epochs, args.max_steps, bf16=not args.fp32)
    if rows:
        print("\n" + chart(rows))

    before = after = []
    if not args.no_eval:
        model = trainer.model
        model.eval()
        before = stage_before_after(tok, model, records, "训练前", args.gen_tokens)
        after = stage_before_after(tok, model, records, "训练后", args.gen_tokens)

    meta = {
        "模型": args.model,
        "数据": f"{args.data}（{len(records)} 条）",
        "epochs": args.epochs,
        "耗时(s)": round(trainer.state.log_history[-1].get("train_runtime", 0), 1)
        if trainer.state.log_history
        else "n/a",
    }
    write_report(meta, before, after, rows)


if __name__ == "__main__":
    main()
