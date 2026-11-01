"""
公共件：路径、打印、tokenizer / 设备小工具。

路径全部以「仓库根目录」为基准（code/ 的上两级），所以命令在仓库根目录执行，
不用 cd 来 cd 去。
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
STEP_DIR = CODE_DIR.parent
ROOT = STEP_DIR.parent

DATA_DIR = STEP_DIR / "data"
RUN_DIR = STEP_DIR / "runs" / "distill"
CKPT_DIR = Path(os.environ.get("CKPT_DIR") or (STEP_DIR / "ckpt" / "distill-agent"))  # CKPT_DIR=... 可覆盖（磁盘紧时指到数据盘）
LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))   # 非交互终端每 N 步打一行
LOCAL_JSONL = DATA_DIR / "distill_prompts.jsonl"

STUDENT = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TEACHER = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


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


def tid_of(tok, text: str) -> int:
    enc = tok(text, add_special_tokens=False)["input_ids"]
    return enc[0] if enc else 0


def ids_of(tok, messages, **kw):
    """transformers 4.x 返回 list、5.x 返回 dict，统一成 list。"""
    out = tok.apply_chat_template(messages, tokenize=True, **kw)
    return list(out["input_ids"]) if hasattr(out, "keys") else list(out)


def device_of(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype_of(torch):
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def version_of(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return "(未安装 —— --inspect 也能跑)"


def print_env(student: str, teacher: str) -> None:
    hr("阶段 0 / 5   环境")
    try:
        import torch

        torch_v = torch.__version__
        cuda = torch.cuda.is_available()
    except ImportError:
        torch_v, cuda = "(未安装)", False
    kv(
        python=sys.version.split()[0],
        transformers=version_of("transformers"),
        trl=version_of("trl"),
        torch=f"{torch_v}  cuda={cuda}",
        student=student,
        teacher=teacher,
    )


# ---------------- 画图 / 实时进度（纯标准库，不依赖 matplotlib） ----------------
import time as _time


def spark(values, width: int = 28) -> str:
    """把一串数压成一行方块字符，用于终端里的实时走势。"""
    if not values:
        return ""
    vals = list(values)[-width:]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(7, int((v - lo) / span * 7.999))] for v in vals)


def fmt_eta(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def plot_svg(path, series, title="", xlabel="step", ylabel="loss",
             width=760, height=310, y_min=None, y_max=None):
    """用纯字符串拼一张折线图（SVG），不需要 matplotlib。

    series: [{"name": "loss", "pts": [(x, y), ...], "color": "#2563eb",
              "width": 1.6, "alpha": 1.0, "dash": None}]
    """
    pad_l, pad_r, pad_t, pad_b = 58, 20, 34, 46
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    xs = [x for s in series for x, _ in s["pts"]]
    ys = [y for s in series for _, y in s["pts"]]
    if not xs:
        return None
    x0, x1 = min(xs), max(xs)
    y0 = min(ys) if y_min is None else y_min
    y1 = max(ys) if y_max is None else y_max
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1.0
    m = (y1 - y0) * 0.06
    y0, y1 = y0 - m, y1 + m

    def sx(x):
        return pad_l + (x - x0) / (x1 - x0) * iw

    def sy(y):
        return pad_t + ih - (y - y0) / (y1 - y0) * ih

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif,system-ui,Segoe UI,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    if title:
        out.append(f'<text x="{pad_l}" y="20" font-size="13" fill="#111827" font-weight="600">{title}</text>')
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4
        py = sy(yv)
        out.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{pad_l + iw}" y2="{py:.1f}" stroke="#eef0f3" stroke-width="1"/>')
        out.append(f'<text x="{pad_l - 8}" y="{py + 4:.1f}" font-size="11" fill="#6b7280" text-anchor="end">{yv:.2f}</text>')
    for i in range(5):
        xv = x0 + (x1 - x0) * i / 4
        px = sx(xv)
        out.append(f'<text x="{px:.1f}" y="{height - 24}" font-size="11" fill="#6b7280" text-anchor="middle">{xv:.0f}</text>')
    out.append(f'<line x1="{pad_l}" y1="{pad_t + ih}" x2="{pad_l + iw}" y2="{pad_t + ih}" stroke="#9ca3af"/>')
    out.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ih}" stroke="#9ca3af"/>')
    out.append(f'<text x="{pad_l + iw / 2:.0f}" y="{height - 6}" font-size="11" fill="#6b7280" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="16" y="{pad_t - 8}" font-size="11" fill="#6b7280">{ylabel}</text>')
    for s in series:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s["pts"])
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        out.append(
            f'<polyline fill="none" stroke="{s.get("color", "#2563eb")}" stroke-width="{s.get("width", 1.6)}" '
            f'stroke-opacity="{s.get("alpha", 1.0)}" stroke-linejoin="round"{dash} points="{pts}"/>'
        )
    # 图例：右上角，避开左边的数据
    lx = pad_l + iw - 128
    ly = pad_t + 12
    for s in series:
        out.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" '
            f'stroke="{s.get("color", "#2563eb")}" stroke-width="{s.get("width", 2.2)}"/>'
        )
        out.append(f'<text x="{lx + 28}" y="{ly + 4}" font-size="11" fill="#374151">{s["name"]}</text>')
        ly += 16
    out.append("</svg>")
    text = "\n".join(out)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return str(path)

def train_runtime(trainer):
    """从 trainer 里取总训练耗时（秒）。不同版本放的位置不一样，能取到就用。"""
    metrics = getattr(trainer, "metrics", None) or {}
    if "train_runtime" in metrics:
        return float(metrics["train_runtime"])
    for rec in reversed(getattr(trainer.state, "log_history", []) or []):
        if "train_runtime" in rec:
            return float(rec["train_runtime"])
    return 0.0


# 训练完成后的落盘：失败必须说清楚，不能让后面的评估/报告把权重一起带走。
def safe_save(trainer, out_dir) -> bool:
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out_dir))
        print(f"  权重已存：{out_dir}")
        return True
    except Exception as e:          # noqa: BLE001 - 这里就是要抓住一切
        print(f"  ⚠ 保存权重失败：{type(e).__name__}: {e}")
        print("    80% 是磁盘满或目录不可写。先看：df -h  /  du -sh ~/.cache/*")
        print("    腾出空间后可以只补存这一步（不用重训）：")
        print(f"      CKPT_DIR=<新路径> python <本脚本> --inspect   # 只是为了确认路径")
        print("    或者直接把环境变量 CKPT_DIR 指到数据盘再重跑。")
        return False


def safe_run(name: str, fn, *a, **kw):
    """跑一个"训练之后"的步骤；挂了只报警，不拖垮整条流水线。"""
    try:
        return fn(*a, **kw)
    except Exception as e:          # noqa: BLE001
        print(f"  ⚠ {name} 失败（已保存的权重/日志不受影响）：{type(e).__name__}: {e}")
        return None


def csv_append(path: Path, row: dict) -> None:
    """把一步的指标追加到 csv（训练中途崩了也能留下历史，不用从头再跑）。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as fh:
            if not new:                      # 沿用第一行写下的表头
                with path.open(encoding="utf-8") as rh:
                    fieldnames = next(csv.reader(rh), list(row.keys()))
            else:
                fieldnames = list(row.keys())
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
            fh.flush()
    except Exception as e:      # 落盘失败不该拖垮训练
        print(f"  ⚠ 写 {path.name} 失败：{e}")
