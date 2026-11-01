"""
报告：终端 ASCII 图 + runs/sft/report.md。

report.md 是给你回传给教程用的——里面已经有环境、前后 held-out 指标、
损失表、损失图、生成对比。跑完直接把这个文件发回来就行。
"""

from __future__ import annotations

import json

from common import RUN_DIR


def chart(rows, width=46) -> str:
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


def write_report(meta, before, after, rows) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# SFT 运行报告", ""]
    for k, v in meta.items():
        md.append(f"- **{k}**：{v}")
    md += ["", "## 损失", "", "| step | loss | lr | grad_norm | token_acc |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['step']} | {r['loss']:.4f} | {r['lr']:.2e} | {r['grad_norm']:.3f} | {r['tok_acc']:.3f} |")
    if rows:
        md += ["", "```", chart(rows), "```"]
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
