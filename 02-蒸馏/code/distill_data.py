"""
阶段 1：读 prompt。

蒸馏的数据和 SFT 长得完全不一样：**只有 prompt，没有答案**。
答案由学生当场写（on-policy），所以这里能打印的东西就是「问题本身」。
"""

from __future__ import annotations

import json
from pathlib import Path

from common import LOCAL_JSONL, hr, kv


def read_prompts(path: Path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_prompt(x):
    """HF 数据集里 prompt 可能是「消息列表」也可能是「一句字符串」，统一成消息列表。"""
    if isinstance(x, str):
        return [{"role": "user", "content": x}]
    return [dict(m) for m in x]


def load_prompts(args):
    """
    返回 (训练集, 评测集, 来源)。

    --dataset 用官方 prompt-only 数据集（`trl-lib/ultrafeedback-prompt`，6.4 万条，
    列名就叫 prompt，正是蒸馏要的形状）；否则读本地 3 条解剖样本。
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
        rows = [{"prompt": normalize_prompt(r["prompt"])} for r in ds]
        src = f"{args.dataset}（取前 {len(rows)} 条）"
    else:
        path = Path(args.data or LOCAL_JSONL)
        rows = read_prompts(path)
        src = f"{path}（本地解剖样本）"

    n_eval = min(args.eval_size, max(1, len(rows) // 10)) if len(rows) >= 4 else 0
    return rows[n_eval:], rows[:n_eval], src


def stage_read(rows, tok, src="", n_eval: int = 0) -> None:
    hr("阶段 1 / 5   读数据（只有 prompt，没有答案）")
    print(f"  来源：{src or LOCAL_JSONL}")
    print(f"  训练 {len(rows)} 条    held-out 评测 {n_eval} 条")
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
