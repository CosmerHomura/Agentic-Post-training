"""
报告：loss.svg 曲线 + runs/distill/report.md。

报告里三块：训练前的分布差（逐 token 表）、训练 loss、held-out 前后对比。
跑完把这个文件发回来，教程里的「示意值」就能换成你的真实数字。
"""

from __future__ import annotations

import json

from common import RUN_DIR


def write_report(payload: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(payload, (RUN_DIR / "report.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

    L = []
    L.append("# 蒸馏（on-policy GKD）运行报告\n")
    L.append(f"- 学生：`{payload['student']}`")
    L.append(f"- 教师：`{payload['teacher']}`")
    L.append(f"- β = {payload['beta']}，temperature = {payload['temperature']}")
    L.append(f"- 生成长度上限：{payload['gen_tokens']} tokens")
    L.append(f"- 样本数：{payload['n_samples']}（评测 {payload.get('n_eval', 0)}），训练步数：{payload['n_steps']}")
    L.append(f"- 数据来源：{payload.get('src', '本地 jsonl')}\n")

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
    L.append("\n![loss 曲线](loss.svg)\n")
    L.append("（训练结束时生成；完整数据见 `loss.csv`）\n")

    L.append("## 3. held-out 评测（训练前 vs 训练后）\n")
    if payload.get("pre") and payload.get("post"):
        L.append("| | 平均前向 KL（学生 vs 教师，越小越像） |")
        L.append("|---|---|")
        L.append(f"| 训练前 | **{payload['pre']['kl']:.4f}** |")
        L.append(f"| 训练后 | **{payload['post']['kl']:.4f}** |")
        for tag in ("pre", "post"):
            for o in payload[tag]["samples"]:
                L.append(f"\n**{tag}** · prompt：{o['prompt'][:80]}\n\n```\n{o['out']}\n```")
    else:
        L.append("（没有 held-out 评测：样本太少或用了 --no-eval）")

    md = RUN_DIR / "report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"  报告已写入：{md}")
