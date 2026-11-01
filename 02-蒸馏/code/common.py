"""
公共件：路径、打印、tokenizer / 设备小工具。

路径全部以「仓库根目录」为基准（code/ 的上两级），所以命令在仓库根目录执行，
不用 cd 来 cd 去。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
STEP_DIR = CODE_DIR.parent
ROOT = STEP_DIR.parent

DATA_DIR = STEP_DIR / "data"
RUN_DIR = ROOT / "runs" / "distill"
CKPT_DIR = ROOT / "ckpt" / "distill-agent"
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
