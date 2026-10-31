"""
第 3 步 · 强化学习（GRPO）—— 全流程可视化版

数据里只有 prompt 和「怎么判对错」，没有答案轨迹。
模型自己写 G 条，奖励函数逐条打分，组内比较算出「谁比平均水平好」，
然后提高好答案的似然、压低差答案的似然。

一条样本进训练，发生这些事：

    1) 读 JSONL            —— prompt + ground_truth（判分用），没有 completion
    2) 套模板 + tokenize    —— 和 SFT 前半段一模一样
    3) 采 G 条 completion   —— 同一个 prompt，采样 8 次
    4) 逐条打分             —— reward_funcs 返回 [r_1..r_G]
    5) 组内算优势           —— A_i = (r_i - mean) / (std + eps)
    6) 算策略梯度损失        —— ratio·A 裁剪后的最小值，且只在 completion 段算
    7) 训练、奖励曲线、前后对比

用法
----
    python train_grpo.py --inspect        # 只跑 1~2 步（只要 tokenizer）
    python train_grpo.py --explain-loss   # 不加载模型，用玩具数字手算一遍 GRPO 损失
    python train_grpo.py                  # 完整流程
    python train_grpo.py --max-steps 30 --num-generations 4

产物
----
    runs/grpo/report.md     ← 把这份发我，我回填教程里的示例输出
    runs/grpo/report.json
    runs/grpo/loss.csv
    ckpt/grpo-agent/

数据
----
    ../data/grpo_prompts.jsonl   每行 {"prompt": [...], "ground_truth": "1081"}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "grpo_prompts.jsonl"
RUN_DIR = HERE.parent / "runs" / "grpo"
CKPT_DIR = HERE.parent / "ckpt" / "grpo-agent"

MODEL = os.environ.get("GRPO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
FORMAT_RE = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.S)

_PRINTED = {"rewards": False}


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
    s = tok.decode([tid], skip_special_tokens=False)
    return s.replace(" ", "␣").replace("\n", "\\n").replace("\t", "\\t")


def ids_of(tok, messages, **kw):
    out = tok.apply_chat_template(messages, tokenize=True, **kw)
    return list(out["input_ids"]) if hasattr(out, "keys") else list(out)


def read_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


INSTRUCTION = (
    "先在心里推理，再给出答案。\n"
    "输出格式必须是：\n"
    "<think>……推理过程……</think>\n"
    "<answer>最终答案（只写数字）</answer>\n\n"
)


def norm_ans(s: str) -> str:
    """把答案归一化：去空白/逗号/美元符，去掉尾随的 .0，统一小写。大小写与格式不该决定对错。"""
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s.lower()


def answer_of(row) -> str:
    """官方数学数据集里答案可能在 answer / 也可能埋在 solution 的 `#### 42` 里。"""
    if row.get("answer") not in (None, ""):
        return norm_ans(row["answer"])
    sol = str(row.get("solution", ""))
    m = re.search(r"####\s*([-\d.,]+)", sol)
    if m:
        return norm_ans(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", sol)
    return norm_ans(nums[-1]) if nums else ""


def load_rows(args):
    """
    返回 (训练集, 评测集, 来源)。

    --dataset 用官方可验证数据集（`trl-lib/DeepMath-103K`，prompt 是数学题、
    有标准答案可判对错）；否则读本地 3 道解剖用小题。
    """
    if args.dataset:
        from datasets import load_dataset

        print(f"  从 HF 加载：{args.dataset}（split={args.split}）")
        try:
            ds = load_dataset(args.dataset, split=args.split)
        except Exception as e:
            raise SystemExit(
                f"  加载失败：{e}\n  国内网络先设 HF_ENDPOINT=https://hf-mirror.com；或用 --data 指向本地 jsonl。"
            ) from e
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        rows = []
        for r in ds:
            prob = r.get("problem") or (r.get("prompt") if isinstance(r.get("prompt"), str) else None)
            if not prob:
                continue
            rows.append(
                {
                    "prompt": [{"role": "user", "content": INSTRUCTION + str(prob).strip()}],
                    "ground_truth": answer_of(r),
                }
            )
        rows = [r for r in rows if r["ground_truth"]]
        src = f"{args.dataset}（取前 {len(rows)} 条）"
    else:
        rows = read_rows(Path(args.data))
        rows = [dict(r, ground_truth=norm_ans(r["ground_truth"])) for r in rows]
        src = str(args.data)

    n_eval = min(args.eval_size, max(1, len(rows) // 10)) if len(rows) >= 4 else 0
    return rows[n_eval:], rows[:n_eval], src


# ============================================================ 奖励函数
def format_reward(completions, **kwargs):
    """格式分：必须 <think>…</think><answer>…</answer>。返回 list[float]，每条一个。"""
    scores = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, (list, tuple)) else comp
        scores.append(1.0 if FORMAT_RE.match(content.strip()) else 0.0)
    _print_rewards("format", scores)
    return scores


def accuracy_reward(completions, ground_truth=None, **kwargs):
    """正确性分：从 <answer> 里抠数字，和 ground_truth 比（先归一化，避免 1,000 与 1000 判错）。"""
    scores = []
    for comp, gt in zip(completions, ground_truth or [None] * len(completions)):
        content = comp[0]["content"] if isinstance(comp, (list, tuple)) else comp
        m = ANSWER_RE.search(content)
        got = norm_ans(m.group(1)) if m else ""
        scores.append(1.0 if got and got == norm_ans(gt) else 0.0)
    _print_rewards("accuracy", scores)
    return scores


def _print_rewards(which: str, scores) -> None:
    """第一次被调用时，把「每个答案拿了多少分」打出来，后面不再刷屏。"""
    if _PRINTED["rewards"] or not os.environ.get("GRPO_SHOW_REWARDS", "1") == "1":
        return
    n = len(scores)
    print(f"  [reward_func] {which} 对 {n} 条 completion 打分：{scores}")
    if which == "accuracy":
        _PRINTED["rewards"] = True


# ============================================================ 1. 读数据
def stage_read(rows, tok, src="", n_eval: int = 0) -> None:
    hr("阶段 1 / 5   读数据（prompt + 判分依据，没有答案轨迹）")
    print(f"  来源：{src or DATA_PATH}")
    print(f"  训练 {len(rows)} 条    held-out 评测 {n_eval} 条")
    print()
    for i, r in enumerate(rows):
        print(f"  ── 第 {i + 1} 条 ──")
        print(f"    prompt      : {r['prompt'][-1]['content'][:70]}…")
        print(f"    ground_truth: {r.get('ground_truth')!r}")
        print(f"    其他列      : {[k for k in r if k not in ('prompt', 'ground_truth')] or '无'}")
        print()
    print("  和 SFT 的区别：这里没有 assistant 轨迹，只有「什么算对」。")
    print("  模型写什么由它自己决定，奖励函数只负责说「好/不好」。")
    print()
    kv(词表大小=len(tok), tokenizer=type(tok).__name__)


# ============================================================ 3.5 held-out 评测
def pass_at_1(tok, model, eval_rows, max_new: int, temperature: float = 0.7):
    """
    held-out 上每题采 1 条，判对错 —— 这就是 pass@1。

    这才是「有没有学会」的硬指标：loss 降了但 pass@1 没动，说明模型只是变得更
    自信，不是变得更对。
    """
    import torch

    dev = model.device
    ok, fmt, samples = 0, 0, []
    for r in eval_rows:
        ids = ids_of(tok, r["prompt"], add_generation_prompt=True)
        t = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model.generate(
                t, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0][len(ids):], skip_special_tokens=True)
        m = ANSWER_RE.search(text)
        got = norm_ans(m.group(1)) if m else ""
        ok += int(bool(got) and got == norm_ans(r["ground_truth"]))
        fmt += int(bool(FORMAT_RE.match(text.strip())))
        if len(samples) < 2:
            samples.append({"gt": r["ground_truth"], "got": got, "out": text.strip()[:400]})
    n = max(1, len(eval_rows))
    return {"pass1": ok / n, "fmt": fmt / n, "n": len(eval_rows), "samples": samples}


# ============================================================ 2. 模板 + tokenize
def stage_template(rows, tok) -> list[int]:
    hr("阶段 2 / 5   套模板 + tokenize（和 SFT 前半段完全一样）")
    msgs = rows[0]["prompt"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"  渲染结果：{len(text)} 个字符")
    print("  " + "-" * 74)
    for line in text.split("\n"):
        print("  " + line.replace("<|im_start|>", "▸ <|im_start|>"))
    print("  " + "-" * 74)

    ids = ids_of(tok, msgs, add_generation_prompt=True)
    print(f"\n  prompt token 数：{len(ids)}；最后三个：{[piece(tok, t) for t in ids[-3:]]}")
    print()
    print("  到这里和 SFT 一模一样。**差别从下一步开始**：")
    print("  SFT 往这个位置塞一条人写的轨迹；GRPO 让模型自己写 G 条。")
    return ids


# ============================================================ 3+4. 采样与打分
def stage_sample_and_score(rows, tok, num_gen: int, temperature: float, max_new: int):
    hr(f"阶段 3 / 5   同一个 prompt 采 {num_gen} 条，然后逐条打分")
    import torch
    from transformers import AutoModelForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()

    row = rows[0]
    ids = ids_of(tok, row["prompt"], add_generation_prompt=True)
    t = torch.tensor([ids], device=dev)

    print(f"  模型：{MODEL}    temperature={temperature}    G={num_gen}")
    print("  注意：这 G 条是**各自独立采样**的，同一条 prompt 会有对有错 —— 这正是 GRPO 需要的。")
    print()

    completions = []
    for i in range(num_gen):
        with torch.no_grad():
            out = model.generate(
                t, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0][len(ids):], skip_special_tokens=True)
        completions.append({"role": "assistant", "content": text})
        body = text.replace("\n", "\n      ")
        print(f"  ── 第 {i + 1} 条 ──")
        print("      " + body[:400])
        print()

    msgs = [[c] for c in completions]
    f_scores = format_reward(msgs)
    a_scores = accuracy_reward(msgs, ground_truth=[row["ground_truth"]] * num_gen)

    total = [f + a for f, a in zip(f_scores, a_scores)]
    print()
    print("  打分汇总（reward = format + accuracy）：")
    print(f"  {'i':>3}  {'format':>7}  {'accuracy':>9}  {'reward':>7}")
    print("  " + "-" * 36)
    for i, (f, a) in enumerate(zip(f_scores, a_scores)):
        print(f"  {i + 1:>3}  {f:>7.1f}  {a:>9.1f}  {f + a:>7.1f}")

    mean = sum(total) / len(total)
    var = sum((x - mean) ** 2 for x in total) / len(total)
    std = var ** 0.5
    adv = [(x - mean) / (std + 1e-4) for x in total]

    print()
    print(f"  组内均值 mean = {mean:.3f}   标准差 std = {std:.3f}")
    print()
    print("  优势 A_i = (r_i - mean) / (std + eps)：")
    print(f"  {'i':>3}  {'reward':>7}  {'A_i':>8}  含义")
    print("  " + "-" * 52)
    for i, (r, a) in enumerate(zip(total, adv)):
        tag = "比平均好 → 提高它的概率" if a > 0.05 else ("比平均差 → 压低" if a < -0.05 else "≈平均 → 不动")
        print(f"  {i + 1:>3}  {r:>7.1f}  {a:>8.3f}  {tag}")

    if std < 1e-3:
        print()
        print("  ⚠️ std ≈ 0：这一组全对或全错，A 全是 0，这一步学不到任何东西。")
        print("     这就是为什么「任务太难或太简单」都训不动，要选模型能做对一部分的题。")
    return total, mean, std, adv, completions


# ============================================================ 手算损失
def explain_loss() -> None:
    hr("手算一遍：GRPO 的损失到底在算什么（不加载模型）")
    print("  假设一次采样 4 条，奖励分别是 1、0、1、0（对/错），G = 4。")
    rewards = [1.0, 0.0, 1.0, 0.0]
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    adv = [(r - mean) / (std + 1e-4) for r in rewards]
    print(f"\n  ① 组内归一化：mean = {mean:.2f}，std = {std:.2f}")
    print(f"     A = {[round(a, 3) for a in adv]}")
    print("     → 对的两条各 +1，错的两条各 -1（正负对称，这是 GRPO 省掉 critic 的关键）")

    print("\n  ② 对某一个 token，看新旧策略的概率比：")
    print("     设某条**正确**答案（A=+1）的某个 token：")
    print("       旧策略 p_old = 0.20，新策略 p_new = 0.26  → ratio = p_new/p_old = 1.30")
    ratio = 0.26 / 0.20
    print(f"       未裁剪项 ratio·A        = {ratio:.2f} × 1 = {ratio:.2f}")
    print(f"       裁剪项 clip(0.8,1.2)·A  = 1.20 × 1 = 1.20")
    print(f"       损失贡献 = -min({ratio:.2f}, 1.20) = -1.20")
    print("     → 概率涨到 1.2 倍以后就不再奖励了，防止一步走太远")
    print()
    print("     若这条是**错误**答案（A=-1），同样 p_new 从 0.20 涨到 0.26：")
    print(f"       ratio·A = 1.30 × (-1) = -1.30；clip 后 = 1.20 × (-1) = -1.20")
    print(f"       损失贡献 = -min(-1.30, -1.20) = +1.30")
    print("     → 错误答案的概率被推高时，损失反而变大，梯度把它压回去")

    print("\n  ③ 最后对这个 batch 里**所有 completion 的 token** 取平均：")
    print("     L = -1/(G·T) Σ_i Σ_t min(ratio·A_i, clip(ratio,1±ε)·A_i) + β·KL")
    print("     注意：prompt 段不参与（mask 掉），只有模型自己生成的 token 算。")
    print("     本教程默认 β = 0.0（不挂参考模型 KL），省一份显存。")


# ============================================================ 5. 训练
def train(rows, epochs, max_steps, num_gen, temperature, max_new, before_hook=None):
    hr("阶段 5 / 5   训练（每个 logging 步打印 loss / reward / 截断率）")
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    ds = Dataset.from_list([{"prompt": r["prompt"], "ground_truth": r["ground_truth"]} for r in rows])

    class LossPrinter(TrainerCallback):
        def __init__(self):
            self.rows = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            r = {
                "step": state.global_step,
                "loss": round(float(logs["loss"]), 4),
                "reward": round(float(logs.get("reward", 0) or 0), 4),
                "reward_std": round(float(logs.get("reward_std", 0) or 0), 4),
                "kl": round(float(logs.get("kl", 0) or 0), 4),
                "len": round(float(logs.get("completions/mean_length", 0) or 0), 1),
                "clip": round(float(logs.get("completions/clipped_ratio", 0) or 0), 3),
            }
            self.rows.append(r)
            print(
                f"  step {r['step']:>3}  loss {r['loss']:.4f}  reward {r['reward']:.3f} "
                f"(std {r['reward_std']:.3f})  kl {r['kl']:.4f}  len {r['len']:.0f}  clip {r['clip']:.2f}"
            )

    printer = LossPrinter()
    has_cuda = torch.cuda.is_available()
    args = GRPOConfig(
        output_dir=str(CKPT_DIR),
        model_init_kwargs={"dtype": "bfloat16" if has_cuda else "float32"},
        use_cpu=not has_cuda,
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",
        num_generations=num_gen,          # ★ G：每个 prompt 采几条
        temperature=temperature,          # ★ 采样温度
        max_completion_length=max_new,    # ★ 单条答案的 token 上限
        beta=0.0,                         # ★ 0 = 不挂参考模型 KL
        loss_type="dapo",                 # ★ TRL 1.16 默认
        scale_rewards="group",            # ★ 组内归一化 → 就是 A_i 那个公式
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps else -1,
        per_device_train_batch_size=num_gen,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        warmup_steps=3,
        lr_scheduler_type="cosine",
        bf16=has_cuda,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,      # reward 要用数据集里的 ground_truth 列
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[format_reward, accuracy_reward],
        args=args,
        train_dataset=ds,
        callbacks=[printer],
    )
    print(f"  模型参数：{sum(p.numel() for p in trainer.model.parameters()) / 1e6:.0f}M")
    print(f"  有效 batch：{num_gen} 条/prompt × 2 步累积")
    print()

    pre = {}
    if before_hook is not None:
        trainer.model.eval()
        pre = before_hook(trainer.model)
        print(f"  训练前 held-out pass@1：{pre['pass1']:.1%}   格式合规：{pre['fmt']:.1%}")

    print()
    result = trainer.train()
    trainer.save_model(args.output_dir)

    post = {}
    if before_hook is not None:
        trainer.model.eval()
        post = before_hook(trainer.model)
        print(f"  训练后 held-out pass@1：{post['pass1']:.1%}   格式合规：{post['fmt']:.1%}")

    print()
    print(f"  训练结束：{result.metrics}")
    return trainer, printer.rows, pre, post


# ============================================================ 奖励曲线
def chart(rows, key: str = "reward", width: int = 40) -> str:
    if not rows:
        return "  （没有记录）"
    vals = [r[key] for r in rows]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    out = [f"  {key} 走势 (min {lo:.3f} / max {hi:.3f})"]
    for v in vals[:width]:
        bar = int(round((v - lo) / span * 30))
        out.append(f"  {v:6.3f} |" + " " * bar + "●")
    out.append("         +" + "-" * 34)
    out.append(f"          step 1 → {len(vals)}")
    return "\n".join(out)


# ============================================================ 报告
def write_report(p: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(p, (RUN_DIR / "report.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

    L = ["# GRPO 运行报告\n"]
    L.append(f"- 模型：`{p['model']}`")
    L.append(f"- G = {p['num_gen']}，temperature = {p['temperature']}，max_completion_length = {p['max_new']}")
    L.append(f"- 样本数：{p['n_samples']}（评测 {p.get('n_eval', 0)}），训练步数：{p['n_steps']}")
    L.append(f"- 数据来源：{p.get('src', '本地 jsonl')}\n")

    L.append("## 1. 采样与打分（第 1 条 prompt）\n")
    L.append("| i | format | accuracy | reward |")
    L.append("|---|---|---|---|")
    for i, (f, a) in enumerate(zip(p["f_scores"], p["a_scores"])):
        L.append(f"| {i + 1} | {f:.1f} | {a:.1f} | {f + a:.1f} |")
    L.append(f"\n- 组内 mean = {p['mean']:.3f}，std = {p['std']:.3f}")
    L.append(f"- 优势 A = {[round(x, 3) for x in p['adv']]}\n")

    L.append("## 2. 训练曲线\n")
    L.append("| step | loss | reward | reward_std | kl | len | clip |")
    L.append("|---|---|---|---|---|---|---|")
    for r in p["train_rows"]:
        L.append(
            f"| {r['step']} | {r['loss']} | {r['reward']} | {r['reward_std']} | {r['kl']} | {r['len']} | {r['clip']} |"
        )
    L.append("\n```\n" + chart(p["train_rows"], "reward") + "\n```\n")

    L.append("## 3. held-out 评测（训练前 vs 训练后）\n")
    if p.get("pre") and p.get("post"):
        L.append("| | pass@1 | 格式合规 |")
        L.append("|---|---|---|")
        L.append(f"| 训练前 | **{p['pre']['pass1']:.1%}** | {p['pre']['fmt']:.1%} |")
        L.append(f"| 训练后 | **{p['post']['pass1']:.1%}** | {p['post']['fmt']:.1%} |")
        for tag in ("post",):
            for s in p[tag]["samples"]:
                L.append(f"\n**{tag}** · 参考答案 `{s['gt']}`，模型给出 `{s['got']}`\n\n```\n{s['out']}\n```")
    else:
        L.append("（没有 held-out 评测：样本太少）")

    md = RUN_DIR / "report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"  报告已写入：{md}")


# ============================================================ main
_TOK = None


def main() -> None:
    global _TOK, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="只跑数据链路（不需 GPU）")
    ap.add_argument("--explain-loss", action="store_true", help="玩具数字手算 GRPO 损失")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--dataset", default=None, help="HF 数据集名，如 trl-lib/DeepMath-103K")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=2000, help="从官方数据集取前多少条")
    ap.add_argument("--eval-size", type=int, default=100, help="留出多少条做 held-out 评测")
    args = ap.parse_args()
    MODEL = args.model

    if args.explain_loss:
        explain_loss()
        return

    hr("阶段 0 / 5   环境")
    import transformers
    from transformers import AutoTokenizer

    try:
        import torch

        torch_v, cuda = torch.__version__, torch.cuda.is_available()
    except ImportError:
        torch_v, cuda = "(未安装)", False
    kv(
        python=sys.version.split()[0],
        transformers=transformers.__version__,
        trl=_version("trl"),
        torch=f"{torch_v}  cuda={cuda}",
        model=MODEL,
    )

    _TOK = AutoTokenizer.from_pretrained(MODEL)
    rows, eval_rows, src = load_rows(args)

    stage_read(rows, _TOK, src, len(eval_rows))
    stage_template(rows, _TOK)
    if args.inspect:
        print()
        print("  --inspect 模式结束（没有采样、没有训练）。")
        print("  去掉 --inspect 就会：采 G 条 → 打分 → 算优势 → 训练 → 报告。")
        return

    import torch  # noqa: F401  (阶段 3 里才真正用)

    total, mean, std, adv, comps = stage_sample_and_score(
        rows, _TOK, args.num_generations, args.temperature, args.max_new
    )

    msgs = [c["content"] for c in comps]
    f_scores = format_reward([[{"role": "assistant", "content": m}] for m in msgs])
    a_scores = accuracy_reward(
        [[{"role": "assistant", "content": m}] for m in msgs], ground_truth=[rows[0]["ground_truth"]] * len(msgs)
    )

    def hook(model):
        return pass_at_1(_TOK, model, eval_rows, min(args.max_new, 256))

    trainer, train_rows, pre, post = train(
        rows, args.epochs, args.max_steps, args.num_generations, args.temperature, args.max_new,
        before_hook=hook if eval_rows else None,
    )
    print()
    print(chart(train_rows, "reward"))
    print()
    print(chart(train_rows, "loss"))

    write_report(
        {
            "model": MODEL,
            "num_gen": args.num_generations,
            "temperature": args.temperature,
            "max_new": args.max_new,
            "n_samples": len(rows),
            "n_eval": len(eval_rows),
            "src": src,
            "n_steps": len(train_rows),
            "f_scores": f_scores,
            "a_scores": a_scores,
            "mean": mean,
            "std": std,
            "adv": adv,
            "train_rows": train_rows,
            "pre": pre,
            "post": post,
        }
    )


def _version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return "(未安装 —— --inspect 也能跑)"


if __name__ == "__main__":
    main()
