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
    "<answer>最终答案（只写数字）</answer>\n\n"
)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
FORMAT_RE = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.S)


def norm_ans(s: str) -> str:
    """归一化：去空白/逗号/美元符，去掉尾随 .0，统一小写。

    不做这一步，`1,000` 与 `1000`、`0.5` 与 `.50` 会被判错，
    奖励信号就变成了「格式运气」而不是「算得对不对」。
    """
    s = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s.lower()


def answer_of(row) -> str:
    """官方数学数据集里，答案可能叫 answer，也可能埋在 solution 的 `#### 42` 里。"""
    if row.get("answer") not in (None, ""):
        return norm_ans(row["answer"])
    sol = str(row.get("solution", ""))
    m = re.search(r"####\s*([-\d.,]+)", sol)
    if m:
        return norm_ans(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", sol)
    return norm_ans(nums[-1]) if nums else ""


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
            prob = (
                r.get("problem")
                or r.get("question")
                or (r.get("prompt") if isinstance(r.get("prompt"), str) else None)
            )
            if not prob:
                continue
            rows.append(
                {
                    "prompt": [{"role": "user", "content": INSTRUCTION + str(prob).strip()}],
                    "ground_truth": answer_of(r),
                }
            )
        rows = [r for r in rows if r["ground_truth"]]
        if not rows:
            raise SystemExit("  这个数据集里没解析出带标准答案的行，换 --dataset 或检查列名。")
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
