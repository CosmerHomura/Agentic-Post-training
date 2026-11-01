"""
公用：路径、打印、分词小工具。

所有路径都由本文件位置推出来，所以**无论从哪个目录启动**，产物都落在仓库根的
`runs/` 与 `ckpt/` 下——这就是「命令可以按顶层目录写」的实现方式。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# code/ 的上两级 = 仓库根（02-蒸馏 这种顶层目录再往上一层）
ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "grpo"
CKPT_DIR = ROOT / "ckpt" / "grpo-agent"
LOCAL_JSONL = Path(__file__).resolve().parent.parent / "data" / "grpo_prompts.jsonl"

MODEL_DEFAULT = os.environ.get("GRPO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def kv(**kw) -> None:
    for k, v in kw.items():
        print(f"  {k:<20} {v}")


def piece(tok, tid: int) -> str:
    """一个 token 长什么样（用真实解码，不猜字节表）。"""
    s = tok.decode([tid], skip_special_tokens=False)
    return s.replace(" ", "␣").replace("\n", "\\n").replace("\t", "\\t")


def ids_of(tok, messages, **kw):
    """transformers 4.x 返回 list，5.x 返回 dict/BatchEncoding，这里统一成 list[int]。"""
    out = tok.apply_chat_template(messages, tokenize=True, **kw)
    return list(out["input_ids"]) if hasattr(out, "keys") else list(out)


def version_of(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return "(未安装 —— --inspect / --explain-loss 也能跑)"


def print_env(model: str) -> None:
    hr("阶段 0 / 5   环境")
    try:
        import torch

        torch_v, cuda = torch.__version__, torch.cuda.is_available()
    except ImportError:
        torch_v, cuda = "(未安装)", False
    kv(
        python=sys.version.split()[0],
        transformers=version_of("transformers"),
        trl=version_of("trl"),
        torch=f"{torch_v}  cuda={cuda}",
        model=model,
        output_dir=str(CKPT_DIR),
    )
