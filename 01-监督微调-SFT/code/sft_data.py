"""
阶段 1：把数据读进来，并切出 held-out 评测集。

两种数据源：
  --dataset trl-lib/Capybara   官方对话数据（1.6 万条），能训出效果
  默认                         仓库里的 2 条解剖样本，只用来看清数据长什么样
"""

from __future__ import annotations

import json
from pathlib import Path

from common import LOCAL_JSONL, hr


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
        recs = [{"messages": r["messages"], "tools": r.get("tools") or None} for r in ds]
        if not any(r["tools"] for r in recs):   # 整批都没工具定义（Capybara 这种纯对话）
            for r in recs:
                r.pop("tools", None)
        src = f"{args.dataset}（取前 {len(recs)} 条）"
    else:
        path = Path(args.data or LOCAL_JSONL)
        recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        src = f"{path}（本地解剖样本）"

    n_eval = min(args.eval_size, max(1, len(recs) // 10)) if len(recs) >= 4 else 0
    return recs[n_eval:], recs[:n_eval], src


def stage_read(src, records, eval_records):
    hr("阶段 1 / 4   读数据")
    print(f"  来源：{src}")
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
    for t in records[0].get("tools") or []:
        f = t["function"]
        print(f"    - {f['name']}({', '.join(f['parameters'].get('properties', {}))})  # {f['description']}")
    if not records[0].get("tools"):
        print("    （这条没有工具定义 —— 官方 Capybara 就是纯对话，想要 agent 轨迹得自己采集）")
    return records


def answer_span(rec) -> int:
    """最后一条 assistant 的 content 长度，用来在评测里判断「模型有没有学会那种格式」。"""
    for m in reversed(rec["messages"]):
        if m["role"] == "assistant":
            return len(m.get("content") or "")
    return 0
