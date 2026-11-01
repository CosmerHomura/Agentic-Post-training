"""
报告：loss.svg 曲线 + runs/sft/report.md。

report.md 是给你回传给教程用的——里面已经有环境、前后 held-out 指标、
损失表、损失图、生成对比。跑完直接把这个文件发回来就行。
"""

from __future__ import annotations

import json
import math

from common import RUN_DIR


def write_report(meta, before, after, rows) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# SFT 运行报告", ""]
    for k, v in meta.items():
        md.append(f"- **{k}**：{v}")
    md += ["", "## 损失", "", "![loss 曲线](loss.svg)", ""]
    if rows:
        md += ["每 20 步刷新一次；完整逐 step 数据见同目录 `loss.csv`。", ""]
        every = max(1, math.ceil(len(rows) / 40))
        md += ["| step | loss | lr | grad_norm | token_acc |", "|---|---|---|---|---|"]
        for r in rows[::every]:
            md.append(f"| {r['step']} | {r['loss']:.4f} | {r['lr']:.2e} | {r['grad_norm']:.3f} | {r['tok_acc']:.3f} |")
        md += ["", f"（只列每 {every} 步一行，共 {len(rows)} 步）"]
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
