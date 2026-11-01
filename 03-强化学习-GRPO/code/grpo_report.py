"""
报告：loss.svg 曲线（loss + reward）+ runs/grpo/report.md。

跑完把 report.md 发回来，教程里的「示意值」就能换成你的真实数字。
"""

from __future__ import annotations

import json

from common import RUN_DIR


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
    L.append("\n![loss / reward 曲线](loss.svg)\n")
    L.append("（训练结束时生成；完整数据见 `loss.csv`）\n")

    L.append("## 3. held-out 评测（训练前 vs 训练后）\n")
    if p.get("pre") and p.get("post"):
        L.append("| | pass@1 | 格式合规 |")
        L.append("|---|---|---|")
        L.append(f"| 训练前 | **{p['pre']['pass1']:.1%}** | {p['pre']['fmt']:.1%} |")
        L.append(f"| 训练后 | **{p['post']['pass1']:.1%}** | {p['post']['fmt']:.1%} |")
        for s in p["post"]["samples"]:
            L.append(f"\n**训练后** · 参考答案 `{s['gt']}`，模型给出 `{s['got']}`\n\n```\n{s['out']}\n```")
    else:
        L.append("（没有 held-out 评测：样本太少）")

    md = RUN_DIR / "report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"  报告已写入：{md}")
