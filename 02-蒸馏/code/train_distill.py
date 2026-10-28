"""
第 2 步 · 蒸馏（On-policy GKD）—— 全流程可视化版

和 SFT 最大的区别：**没有标准答案**。
数据里只有 prompt，答案由学生当场生成；然后教师逐 token 给出「每个词应该有多大概率」，
学生去对齐这个分布。

一条样本进模型前后，发生这些事：

    1) 只读 prompt            —— 数据里没有 completion
    2) 套模板（add_generation_prompt=True）—— 结尾会留一个 <|im_start|>assistant\\n，等模型写
    3) tokenize               —— 只有 prompt 的 id
    4) 学生自己生成 completion —— 这就是 on-policy：在自己写的东西上学
    5) 教师、学生对「同一个 next token」各给一个分布
       └─ 学生被拉向教师：L = β·KL(p_T‖p_M) + (1-β)·KL(p_S‖p_M)，p_M = (1-β)p_S + βp_T
    6) 训练、画损失、训练前后对比

用法
----
    python train_distill.py --inspect    # 只跑 1~3 步（只要 tokenizer，不需要 GPU）
    python train_distill.py              # 完整流程：生成 → 分布对比 → 训练 → 报告
    python train_distill.py --max-steps 20 --gen-tokens 96

产物
----
    runs/distill/report.md     ← 把这份发我，我回填教程里的示例输出
    runs/distill/report.json
    runs/distill/tokens.txt    全量 token 表
    runs/distill/loss.csv
    ckpt/distill-agent/

数据
----
    ../data/distill_prompts.jsonl   每行 {"prompt": [{"role":"user","content": ...}]}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "distill_prompts.jsonl"
RUN_DIR = HERE.parent / "runs" / "distill"
CKPT_DIR = HERE.parent / "ckpt" / "distill-agent"

STUDENT = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TEACHER = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


# ============================================================ 0. 小工具
def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def kv(**kw) -> None:
    for k, v in kw.items():
        print(f"  {k:<20} {v}")


def piece(tok, tid: int) -> str:
    """token → 可读文本（byte-level BPE 必须用 decode，否则是乱码）。"""
    s = tok.decode([tid], skip_special_tokens=False)
    return s.replace(" ", "␣").replace("\n", "\\n").replace("\t", "\\t")


def ids_of(tok, messages, **kw):
    """transformers 4.x 返回 list、5.x 返回 dict，统一成 list。"""
    out = tok.apply_chat_template(messages, tokenize=True, **kw)
    return list(out["input_ids"]) if hasattr(out, "keys") else list(out)


def read_prompts(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def device_of(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype_of(torch):
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ============================================================ 1. 读数据
def stage_read(rows, tok) -> None:
    hr("阶段 1 / 5   读 JSONL（只有 prompt，没有答案）")
    print(f"  文件：{DATA_PATH}")
    print(f"  行数：{len(rows)}")
    print()
    for i, r in enumerate(rows):
        msgs = r["prompt"]
        print(f"  ── 第 {i + 1} 条  ({len(msgs)} 条消息) ──")
        for m in msgs:
            text = m["content"].replace("\n", "\n     ")
            print(f"    [{m['role']}] {text}")
        print()

    print("  注意：这里**没有** completion / answer / ground_truth 字段。")
    print("  蒸馏的答案不是数据给的，是学生模型当场写的 —— 这就是 on-policy 的含义。")
    print()
    kv(词表大小=len(tok), tokenizer=type(tok).__name__, chat_template=f"{len(tok.chat_template)} 字符")


# ============================================================ 2. 套模板
def stage_template(rows) -> None:
    hr("阶段 2 / 5   套 chat template（JSON → 纯文本）")
    msgs = rows[0]["prompt"]
    tok = _TOK
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"  渲染结果：{len(text)} 个字符（add_generation_prompt=True，末尾多出 assistant 的角色头）")
    print("  " + "-" * 74)
    for line in text.split("\n"):
        print("  " + line.replace("<|im_start|>", "▸ <|im_start|>"))
    print("  " + "-" * 74)
    print()
    print("  三个关键点：")
    print("   1) 结尾的 ▸ <|im_start|>assistant 是**留白**：它在等模型往下写，不是数据内容。")
    print("   2) 蒸馏没有 <tool_response>、没有 <tool_call> —— 因为数据里没有工具调用示例；")
    print("      想让蒸馏学会多轮，得在 prompt 里带 tools，并用 max_tool_calling_iterations 放开。")
    print("   3) 学生与教师必须用同一个 tokenizer，否则「第 7 个 token」指的不是同一个词。")


# ============================================================ 3. tokenize
def stage_tokens(rows) -> list[int]:
    hr("阶段 3 / 5   tokenize 后，模型真正看到的 prompt")
    tok = _TOK
    msgs = rows[0]["prompt"]
    ids = ids_of(tok, msgs, add_generation_prompt=True)
    print(f"  prompt token 数：{len(ids)}")
    print()
    print(f"  {'idx':>4}  {'token_id':>8}  token")
    print("  " + "-" * 44)
    for i, tid in enumerate(ids):
        print(f"  {i:>4}  {tid:>8}  {piece(tok, tid)}")
    print()
    print("  最后几个 token 是模板加的「请开始写」：")
    print(f"    {[piece(tok, t) for t in ids[-3:]]}")
    return ids


# ============================================================ 4. 学生生成
def stage_generate(prompt_ids, tok, gen_tokens: int, temperature: float, seed: int = 0):
    """学生写 completion —— on-policy 的那一步。"""
    hr("阶段 4 / 5   学生自己生成 completion（on-policy）")
    import torch
    from transformers import AutoModelForCausalLM

    dev, dtype = device_of(torch), dtype_of(torch)
    torch.manual_seed(seed)
    student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=dtype).to(dev).eval()

    ids = torch.tensor([prompt_ids], device=dev)
    with torch.no_grad():
        out = student.generate(
            ids,
            max_new_tokens=gen_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen = out[0][len(prompt_ids):].tolist()
    text = tok.decode(gen, skip_special_tokens=False)

    print(f"  学生：{STUDENT}")
    print(f"  采样：temperature={temperature}  max_new_tokens={gen_tokens}")
    print()
    print(f"  生成 {len(gen)} 个 token：")
    print("  " + "-" * 74)
    for line in text.split("\n"):
        print("  " + line)
    print("  " + "-" * 74)
    print()
    print("  这段文字**没有标准答案可以对照**。它是学生自己的尝试，可能对、可能错、")
    print("  可能中途跑偏 —— 教师不改它，只是告诉它「换了我，每个位置会怎么分布」。")
    print()
    print("  completion 的 token 表（前 24 个）：")
    print(f"  {'i':>3}  {'token_id':>8}  token")
    print("  " + "-" * 40)
    for i, tid in enumerate(gen[:24]):
        print(f"  {i:>3}  {tid:>8}  {piece(tok, tid)}")
    if len(gen) > 24:
        print(f"  …  还有 {len(gen) - 24} 个")
    return student, gen


# ============================================================ 5. 分布对比
def stage_distributions(prompt_ids, gen, tok, student, beta: float, temperature: float):
    """
    教师 vs 学生：同一个上下文，各自的 next-token 分布。

    p_T = softmax(teacher_logits / T)，p_S = softmax(student_logits / T)
    p_M = (1-β)·p_S + β·p_T
    L   = β·KL(p_T‖p_M) + (1-β)·KL(p_S‖p_M)
    """
    hr("阶段 5 / 5   教师 vs 学生：逐 token 的分布差多少")
    import torch
    from transformers import AutoModelForCausalLM

    dev, dtype = device_of(torch), dtype_of(torch)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=dtype).to(dev).eval()

    full = prompt_ids + gen
    t = torch.tensor([full], device=dev)
    with torch.no_grad():
        ls = student(t).logits[0]
        lt = teacher(t).logits[0]

    # 第 i 个 completion token 的预测，来自位置 len(prompt)+i-1
    start = len(prompt_ids) - 1
    tmp = temperature

    def softmax(x):
        x = x / tmp
        return torch.softmax(x, dim=-1)

    rows = []
    for i, tid in enumerate(gen):
        pos = start + i
        ps, pt = softmax(ls[pos]), softmax(lt[pos])
        p_tok_s, p_tok_t = float(ps[tid]), float(pt[tid])
        top_t = torch.topk(pt, 3)
        kl_fwd = float(torch.sum(pt * (torch.log(pt + 1e-9) - torch.log(ps + 1e-9))))
        rows.append(
            {
                "i": i,
                "token": tok.decode([tid], skip_special_tokens=False),
                "p_student": p_tok_s,
                "p_teacher": p_tok_t,
                "kl_forward": kl_fwd,
                "teacher_top1": tok.decode([int(top_t.indices[0])], skip_special_tokens=False),
                "teacher_top1_p": float(top_t.values[0]),
            }
        )

    print(f"  教师：{TEACHER}（{sum(p.numel() for p in teacher.parameters()) / 1e6:.0f}M 参数）")
    print(f"  学生：{STUDENT}（{sum(p.numel() for p in student.parameters()) / 1e6:.0f}M 参数）")
    print()
    print("  同一个上下文下，两边对「下一个词」的判断：")
    print(f"  {'i':>3}  {'学生写的词':<12} {'p_学生':>8} {'p_教师':>8} {'教师首选':<12} {'p_首选':>7}")
    print("  " + "-" * 68)
    for r in rows[:14]:
        print(
            f"  {r['i']:>3}  {piece(tok, _tid(tok, r['token']))!s:<12} "
            f"{r['p_student']:>8.3f} {r['p_teacher']:>8.3f} "
            f"{piece(tok, _tid(tok, r['teacher_top1']))!s:<12} {r['teacher_top1_p']:>7.3f}"
        )
    if len(rows) > 14:
        print(f"  …  还有 {len(rows) - 14} 个 token")

    mean_kl = sum(r["kl_forward"] for r in rows) / len(rows)
    # 用 β 混合出 loss（与 TRL 的 GKD 目标一致）
    loss_terms = []
    for i, tid in enumerate(gen):
        pos = start + i
        ps, pt = softmax(ls[pos]), softmax(lt[pos])
        pm = (1 - beta) * ps + beta * pt
        kl_tm = torch.sum(pt * (torch.log(pt + 1e-9) - torch.log(pm + 1e-9)))
        kl_sm = torch.sum(ps * (torch.log(ps + 1e-9) - torch.log(pm + 1e-9)))
        loss_terms.append(float(beta * kl_tm + (1 - beta) * kl_sm))
    loss = sum(loss_terms) / len(loss_terms)

    print()
    print(f"  平均前向 KL  D_KL(p_T‖p_S) = {mean_kl:.4f}   （越高说明学生离教师越远）")
    print(f"  β = {beta} 时的蒸馏损失        = {loss:.4f}")
    print()
    print("  β 怎么选（同一个公式的三种含义）：")
    print("    β = 0.0  →  L = KL(p_T‖p_S)   前向 KL：学生要覆盖教师的**所有**可能性，容易学到「模糊话」")
    print("    β = 1.0  →  L = KL(p_S‖p_T)   反向 KL：学生只盯教师最确定的那条路，容易「塌缩」（TRL 默认）")
    print("    β = 0.5  →  L = JSD            折中，也是本脚本默认")
    return rows, mean_kl, loss, teacher


def _tid(tok, text: str) -> int:
    enc = tok(text, add_special_tokens=False)["input_ids"]
    return enc[0] if enc else 0


# ============================================================ 6. 训练
def train(rows, epochs: float, max_steps: int | None, beta: float, temperature: float, gen_tokens: int):
    hr("阶段 6   训练（学生每个 logging 步打印一次 loss）")
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import DistillationConfig, DistillationTrainer

    ds = Dataset.from_list([{"prompt": r["prompt"]} for r in rows])

    class LossPrinter(TrainerCallback):
        def __init__(self):
            self.rows = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            self.rows.append(
                {
                    "step": state.global_step,
                    "loss": round(float(logs["loss"]), 4),
                    "lr": float(logs.get("learning_rate", 0.0)),
                    "grad_norm": round(float(logs.get("grad_norm", 0) or 0), 4),
                    "entropy": round(float(logs.get("entropy", 0) or 0), 4),
                    "mean_len": round(float(logs.get("completions/mean_length", 0) or 0), 1),
                    "clip_ratio": round(float(logs.get("completions/clipped_ratio", 0) or 0), 3),
                }
            )
            r = self.rows[-1]
            print(
                f"  step {r['step']:>3}  loss {r['loss']:.4f}  lr {r['lr']:.2e}  "
                f"grad {r['grad_norm']:.3f}  ent {r['entropy']:.2f}  "
                f"len {r['mean_len']:.0f}  clip {r['clip_ratio']:.2f}"
            )

    printer = LossPrinter()
    has_cuda = torch.cuda.is_available()
    args = DistillationConfig(
        output_dir=str(CKPT_DIR),
        model_init_kwargs={"dtype": "bfloat16" if has_cuda else "float32"},
        use_cpu=not has_cuda,
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",
        beta=beta,                      # ★ 用哪种散度
        temperature=temperature,        # ★ 同时作用于生成与损失
        max_completion_length=gen_tokens,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps else -1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,             # ★ 蒸馏默认比 SFT 小
        warmup_steps=3,
        lr_scheduler_type="cosine",
        bf16=has_cuda,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = DistillationTrainer(
        model=STUDENT,
        teacher_model=TEACHER,
        args=args,
        train_dataset=ds,
        callbacks=[printer],
    )
    print(f"  学生参数：{sum(p.numel() for p in trainer.model.parameters()) / 1e6:.0f}M")
    print(f"  教师参数：{sum(p.numel() for p in trainer.teacher_model.parameters()) / 1e6:.0f}M")
    print()
    result = trainer.train()
    trainer.save_model(args.output_dir)
    print()
    print(f"  训练结束：{result.metrics}")
    print(f"  权重已存：{args.output_dir}")
    return trainer, printer.rows


# ============================================================ 7. 损失图
def chart(rows, width: int = 46) -> str:
    if not rows:
        return "  （没有 loss 记录）"
    losses = [r["loss"] for r in rows]
    lo, hi = min(losses), max(losses)
    span = (hi - lo) or 1.0
    lines = [f"  loss 走势 (min {lo:.3f} / max {hi:.3f})"]
    n = max(1, min(width, len(losses)))
    step = max(1, len(losses) // n)
    sample = losses[::step][:n]
    for v in sample:
        bar = int(round((v - lo) / span * 34))
        lines.append(f"  {v:6.3f} |" + " " * bar + "●")
    lines.append("         +" + "-" * 38)
    lines.append(f"          step 1 → {len(losses)}")
    return "\n".join(lines)


# ============================================================ 8. 前后对比
def stage_before_after(rows, trainer, tok, gen_tokens: int, is_cuda: bool):
    hr("阶段 7   训练后：同一个 prompt，学生现在怎么写")
    import torch

    prompts = rows[:2]
    outs = []
    for i, r in enumerate(prompts):
        msgs = r["prompt"]
        ids = ids_of(tok, msgs, add_generation_prompt=True)
        t = torch.tensor([ids], device=trainer.model.device)
        with torch.no_grad():
            out = trainer.model.generate(
                t,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0][len(ids):], skip_special_tokens=False)
        outs.append({"prompt": msgs[-1]["content"], "after": text})
        print(f"  ── 用例 {i + 1} ──")
        print(f"  prompt：{msgs[-1]['content'][:60]}…")
        print(f"  蒸馏后：{text.strip()[:300]}")
        print()
    return outs


# ============================================================ 9. 报告
def write_report(payload: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(payload, (RUN_DIR / "report.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

    L = []
    L.append("# 蒸馏（on-policy GKD）运行报告\n")
    L.append(f"- 学生：`{payload['student']}`")
    L.append(f"- 教师：`{payload['teacher']}`")
    L.append(f"- β = {payload['beta']}，temperature = {payload['temperature']}")
    L.append(f"- 生成长度上限：{payload['gen_tokens']} tokens")
    L.append(f"- 样本数：{payload['n_samples']}，训练步数：{payload['n_steps']}\n")

    L.append("## 1. 训练前：教师与学生的分布差\n")
    L.append(f"- 平均前向 KL：**{payload['mean_kl']:.4f}**")
    L.append(f"- 蒸馏损失：**{payload['loss']:.4f}**\n")
    L.append("| i | 学生写的 | p_学生 | p_教师 | 教师首选 | p_首选 |")
    L.append("|---|---|---|---|---|---|")
    for r in payload["dist_rows"][:20]:
        L.append(
            f"| {r['i']} | `{r['token']}` | {r['p_student']:.3f} | {r['p_teacher']:.3f} "
            f"| `{r['teacher_top1']}` | {r['teacher_top1_p']:.3f} |"
        )

    L.append("\n## 2. 训练 loss\n")
    L.append("| step | loss | lr | grad | entropy | 生成长度 | 截断率 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in payload["train_rows"]:
        L.append(
            f"| {r['step']} | {r['loss']} | {r['lr']:.2e} | {r['grad_norm']} "
            f"| {r['entropy']} | {r['mean_len']} | {r['clip_ratio']} |"
        )
    L.append("\n```\n" + chart(payload["train_rows"]) + "\n```\n")

    L.append("## 3. 训练前后对比\n")
    for i, o in enumerate(payload["before_after"]):
        L.append(f"### 用例 {i + 1}\n")
        L.append(f"**prompt**：{o['prompt']}\n")
        L.append(f"**蒸馏后**：\n\n```\n{o['after']}\n```\n")

    md = RUN_DIR / "report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"  报告已写入：{md}")


# ============================================================ main
_TOK = None


def main() -> None:
    global _TOK, STUDENT, TEACHER
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="只跑数据链路（不需 GPU / 不需 trl）")
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gen-tokens", type=int, default=160)
    args = ap.parse_args()

    STUDENT, TEACHER = args.model, args.teacher

    hr("阶段 0 / 5   环境")
    import transformers
    from transformers import AutoTokenizer

    try:
        import torch

        torch_v = torch.__version__
        cuda = torch.cuda.is_available()
    except ImportError:
        torch_v, cuda = "(未安装)", False
    kv(
        python=sys.version.split()[0],
        transformers=transformers.__version__,
        trl=_version("trl"),
        torch=f"{torch_v}  cuda={cuda}",
        student=STUDENT,
        teacher=TEACHER,
    )

    _TOK = AutoTokenizer.from_pretrained(STUDENT)
    rows = read_prompts(DATA_PATH)

    stage_read(rows, _TOK)
    stage_template(rows)
    prompt_ids = stage_tokens(rows)

    if args.inspect:
        print()
        print("  --inspect 模式结束（没有生成、没有训练）。")
        print("  去掉 --inspect 就会：学生生成 → 教师对比分布 → 训练 → 报告。")
        return

    student, gen = stage_generate(prompt_ids, _TOK, args.gen_tokens, args.temperature)
    dist_rows, mean_kl, loss, _ = stage_distributions(prompt_ids, gen, _TOK, student, args.beta, args.temperature)

    trainer, train_rows = train(rows, args.epochs, args.max_steps, args.beta, args.temperature, args.gen_tokens)
    print()
    print(chart(train_rows))

    before_after = stage_before_after(rows, trainer, _TOK, min(args.gen_tokens, 120), cuda)

    payload = {
        "student": STUDENT,
        "teacher": TEACHER,
        "beta": args.beta,
        "temperature": args.temperature,
        "gen_tokens": args.gen_tokens,
        "n_samples": len(rows),
        "n_steps": len(train_rows),
        "mean_kl": mean_kl,
        "loss": loss,
        "dist_rows": dist_rows,
        "train_rows": train_rows,
        "before_after": before_after,
    }
    write_report(payload)


def _version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return "(未安装 —— --inspect 也能跑)"


if __name__ == "__main__":
    main()
