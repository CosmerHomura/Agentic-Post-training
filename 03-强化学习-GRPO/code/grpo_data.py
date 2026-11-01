"""
阶段 1 · 数据：只有 prompt 和「怎么判对错」。

SFT 的数据是「问题 + 标准答案」，这里换成「问题 + 判分依据」——
模型写什么由它自己决定，奖励函数只负责说好或不好。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from common import LOCAL_JSONL, hr, kv

# 只写数字答案，方便程序判分（人工评分任务不需要这个约束）
INSTRUCTION = (
    "先在心里推理，再给出答案。\n"
    "输出格式必须是：\n"
    "<think>……推理过程……</think>\n"
    "<answer>最终答案</answer>\n\n"
    "answer 里只写最终结果本身：数字就不带逗号和单位，表达式照抄，Yes/No 原样。\n"
)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
FORMAT_RE = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.S)


def norm_ans(s) -> str:
    """归一化，让「人类写法」和「模型写法」能直接比。

    真实数据集里的答案长这样：`58`、`$58$`、`\\boxed{58}`、`\\frac{1}{2}`、`Yes`。
    不统一，`1,000` 与 `1000`、`0.5` 与 `.50` 都会被判错，
    奖励信号就变成「格式运气」而不是「算得对不对」。
    """
    s = str(s).strip()
    s = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)   # \frac{a}{b} -> a/b
    s = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)              # \left \right \, 这类命令
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(",", "").replace(" ", "").replace("\u00a0", "")
    s = s.strip(".\u3002;\uff1b")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    if re.fullmatch(r"-?\d+\.\d+", s):          # 0.50 -> 0.5
        s = f"{float(s):g}"
    return s.lower()


BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
HASH_RE = re.compile(r"####\s*([^\n]+)")


def answer_of(row) -> str:
    """把「标准答案」从各种数据集里挖出来。

    真实数据集的形态差别很大：
      · `trl-lib/DeepMath-103K`：答案就是 `solution` 那一列，可能是 `Yes`、`58`、`\\boxed{\\frac{1}{2}}`
      · GSM8K 风格：答案埋在 `solution` 末尾的 `#### 42`
      · 自己整理的数据：单独一列 `answer` / `ground_truth`
    """
    for key in ("answer", "ground_truth", "final_answer", "target"):
        v = row.get(key)
        if v not in (None, ""):
            return norm_ans(v)

    sol = str(row.get("solution") or "")
    if not sol:
        return ""
    m = BOXED_RE.search(sol)
    if m:
        return norm_ans(m.group(1))
    m = HASH_RE.search(sol)
    if m:
        return norm_ans(m.group(1))
    if len(sol) <= 40 and "\n" not in sol:      # 短答案：整段就是答案
        return norm_ans(sol)
    nums = re.findall(r"-?\d+(?:\.\d+)?", sol)
    return norm_ans(nums[-1]) if nums else norm_ans(sol)


def prompt_of(row):
    """数据集里的 prompt 有两种写法：纯文本，或者 chat 消息列表。

    `trl-lib/DeepMath-103K` 的 prompt 就是 `[{"role": "user", "content": "..."}]`，
    直接 `str()` 会变成一坨 Python repr，必须原样接过来。
    """
    p = row.get("prompt")
    if isinstance(p, list) and p and isinstance(p[0], dict):
        msgs = [
            {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
            for m in p
        ]
        return [{"role": "system", "content": INSTRUCTION.strip()}] + msgs
    text = row.get("problem") or row.get("question") or (p if isinstance(p, str) else None)
    if not text:
        return None
    return [{"role": "user", "content": INSTRUCTION + str(text).strip()}]


def read_jsonl(path: Path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(args):
    """
    返回 (训练集, 评测集, 来源)。

    --dataset 用官方可验证数据集（`trl-lib/DeepMath-103K`：题目 + 标准答案）；
    否则读本地 3 道解剖用小题。前 eval_size 条留作 held-out，不参与训练。
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
            msgs = prompt_of(r)
            if not msgs:
                continue
            rows.append({"prompt": msgs, "ground_truth": answer_of(r)})
        rows = [r for r in rows if r["ground_truth"]]
        if not rows:
            cols = list(ds.features) if hasattr(ds, "features") else list(ds[0].keys())
            sample = ds[0] if len(ds) else {}
            raise SystemExit(
                "  这个数据集里没解析出带标准答案的行。\n"
                f"  列名：{cols}\n"
                f"  第一行：{str(sample)[:300]}\n"
                "  用 --data 指向自己的 jsonl，或换一个数据集。"
            )
        src = f"{args.dataset}（取前 {len(rows)} 条）"
    else:
        rows = read_jsonl(Path(args.data))
        rows = [dict(r, ground_truth=norm_ans(r["ground_truth"])) for r in rows]
        src = str(args.data)

    # 评测集最多占 1/10，且至少留 1 条；样本太少就不评（数字没有意义）
    n_eval = min(args.eval_size, max(1, len(rows) // 10)) if len(rows) >= 4 else 0
    return rows[n_eval:], rows[:n_eval], src


def stage_read(rows, tok, src="", n_eval: int = 0) -> None:
    hr("阶段 1 / 5   读数据（prompt + 判分依据，没有答案轨迹）")
    print(f"  来源：{src or LOCAL_JSONL}")
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
