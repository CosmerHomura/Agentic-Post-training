"""
公共件：路径、打印、tokenizer 小工具。

这个文件和具体步骤无关，后面 02/03 章的 code/ 目录里各有一份同构的副本
（刻意不共享：每个章节的代码都能单独拷走运行，不依赖仓库其他部分）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---- 路径：全部以「仓库根目录」为基准，所以任何地方运行都找得到文件 ----
#      code/ 的上三级 = 仓库根（01-监督微调-SFT/code/common.py → 仓库根）
CODE_DIR = Path(__file__).resolve().parent
STEP_DIR = CODE_DIR.parent
ROOT = STEP_DIR.parent

DATA_DIR = STEP_DIR / "data"
RUN_DIR = ROOT / "runs" / "sft"
CKPT_DIR = ROOT / "ckpt" / "sft-agent"
LOCAL_JSONL = DATA_DIR / "sft_agent_trace.jsonl"


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def kv(**kw) -> None:
    for k, v in kw.items():
        print(f"  {k:<22} {v}")


def piece(tok, tid: int) -> str:
    """
    把 token 还原成可读文本。

    注意：Qwen 用的是 byte-level BPE，convert_ids_to_tokens 给的是「字节映射后的伪 latin 串」，
    直接把那个字符串打出来会是乱码；用 decode 才是人类可读的文本。
    单个 token 可能只是某个汉字的一个字节，decode 出来就是 � —— 这不是 bug，正是 tokenizer 的现实。
    """
    s = tok.decode([tid], skip_special_tokens=False)
    return s.replace(" ", "␣").replace("\n", "\\n").replace("\t", "\\t")


def ids_of(tok, messages, tools, **kw):
    """apply_chat_template 在 transformers 4.x 返回 list、5.x 返回 dict，这里统一成 list。"""
    out = tok.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=False, **kw
    )
    return list(out["input_ids"]) if hasattr(out, "keys") else list(out)


def version_of(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except ImportError:
        return "(未安装)"


def print_env(data_desc: str) -> None:
    """阶段 0：把「跑在什么环境、用什么数据」先摊开，省得后面排查。"""
    hr("阶段 0 / 4   环境")
    print(f"  python        {sys.version.split()[0]}")
    print(f"  transformers  {version_of('transformers')}")
    print(f"  trl           {version_of('trl')}")
    try:
        import torch

        print(f"  torch         {torch.__version__}   cuda={torch.cuda.is_available()}")
    except ImportError:
        print("  torch         (未安装 —— 只能跑 --inspect)")
    print(f"  数据          {data_desc}")
