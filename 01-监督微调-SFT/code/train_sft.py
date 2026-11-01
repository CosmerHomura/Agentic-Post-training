"""
第 1 步 · 监督微调（SFT）—— 入口

一条轨迹进模型之前要变四次形：

    JSONL 记录 --chat template--> 纯文本 --tokenizer--> token 序列
        --掩码--> labels --collate--> 一个 batch 张量

本脚本把每一步都打印出来，再跑训练、画损失、做训练前后对比。

用法（**在仓库根目录执行**，不要 cd 进 code/）
--------------------------------------------
    python 01-监督微调-SFT/code/train_sft.py --inspect
    python 01-监督微调-SFT/code/train_sft.py --explain-loss
    python 01-监督微调-SFT/code/train_sft.py                       # 本地 2 条解剖样本
    python 01-监督微调-SFT/code/train_sft.py --dataset trl-lib/Capybara --limit 2000
    python 01-监督微调-SFT/code/train_sft.py --dataset trl-lib/Capybara --limit 2000 --max-steps 50

代码结构（同目录下 6 个文件，各管一件事）
----------------------------------------
    common.py       路径 / 打印 / tokenizer 小工具
    sft_data.py     读数据 + 切 held-out
    sft_template.py 模板 → token → label/mask → batch（阶段 2~4 的打印都在这）
    sft_trainer.py  训练 / 评测 / 生成对比
    sft_report.py   ASCII 损失图 + report.md
    sft_loss.py     手算交叉熵（--explain-loss）
    train_sft.py    就是你正在看的这个：解析参数、按顺序调上面几个

产物（都在本章目录 01-监督微调-SFT/ 下）
------------------------
    runs/sft/report.md     总报告，跑完发给我
    runs/sft/report.json   机器可读版
    runs/sft/tokens.txt    全量 token/label/mask 表
    runs/sft/loss.csv      每步 loss
    ckpt/sft-agent/        训练好的权重
"""

from __future__ import annotations

import argparse

from common import safe_run, LOCAL_JSONL, RUN_DIR, hr, train_runtime
from sft_data import load_records, stage_read
from sft_loss import explain_loss
from sft_report import write_report
from sft_template import stage_batch, stage_template, stage_tokens
from sft_trainer import stage_before_after, train

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args():
    ap = argparse.ArgumentParser(
        description="SFT 全流程（数据链路可视化 + 训练 + 前后对比）",
        epilog="所有路径以仓库根目录为基准，命令在仓库根目录执行即可。",
    )
    ap.add_argument("--inspect", action="store_true", help="只跑数据链路，不训练（可无 GPU）")
    ap.add_argument("--explain-loss", action="store_true", help="玩具数字手算交叉熵（不需 GPU）")
    ap.add_argument("--data", default=str(LOCAL_JSONL), help="本地 jsonl 路径")
    ap.add_argument("--dataset", default=None, help="HF 数据集名，如 trl-lib/Capybara（优先于 --data）")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=2000, help="从官方数据集取前多少条")
    ap.add_argument("--eval-size", type=int, default=100, help="留出多少条做 held-out 评测")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true", help="跳过训练前后生成对比")
    ap.add_argument("--gen-tokens", type=int, default=220, help="生成对比时最多新生成多少 token")
    ap.add_argument("--tensorboard", action="store_true",
                    help="把日志写进 <章节>/runs/<x>/tb，供 TensorBoard / AutoPanel 查看")
    ap.add_argument("--eval-steps", type=int, default=0,
                    help="训练中每 N 步在 held-out 上评估一次（0 = 每 epoch 一次）")
    ap.add_argument("--plot-every", type=int, default=0, help="每多少步重画一次 loss.svg（默认 0 = 只在结束时画一次）")
    ap.add_argument("--fp32", action="store_true", help="不用 bf16")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.explain_loss:
        explain_loss()
        return

    from common import print_env

    print_env(args.dataset or args.data)

    records, eval_records, src = load_records(args)
    stage_read(src, records, eval_records)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"\n  tokenizer     {type(tok).__name__}  vocab={len(tok)}  "
          f"chat_template={len(tok.chat_template or '')} 字符")

    # 解剖：始终用第一条训练样本，保证「模板/token/掩码」讲的是同一份数据
    stage_template(tok, records[0])
    stage_tokens(tok, records[0])
    stage_batch(tok, records[:2])

    if args.inspect:
        print("\n  --inspect 模式结束（没有训练）。去掉 --inspect 就跑完整流程。")
        return

    def grab_samples(model):
        model.eval()
        return stage_before_after(tok, model, eval_records, "生成", args.gen_tokens)

    trainer, rows, pre, post, before, after = train(
        args.model,
        records,
        [] if args.no_eval else eval_records,
        args.epochs,
        args.max_steps,
        bf16=not args.fp32,
        plot_every=args.plot_every,
        tb=args.tensorboard,
        eval_steps=args.eval_steps,
        before_hook=None if args.no_eval else grab_samples,
    )
    if rows:
        print(f"\n  曲线图：{RUN_DIR / 'loss.svg'}")
    if pre and post:
        d = pre["eval_loss"] - post["eval_loss"]
        print(f"\n  held-out -ln p：{pre['eval_loss']:.4f} → {post['eval_loss']:.4f}"
              f"（{'下降' if d > 0 else '上升'} {abs(d):.4f}）")

    runtime = train_runtime(trainer)
    meta = {
        "模型": args.model,
        "数据": f"{src}（训练 {len(records)} 条 / 评测 {len(eval_records)} 条）",
        "epochs": args.epochs,
        "held-out -ln p（训练前）": f"{pre['eval_loss']:.4f}" if pre else "未评测",
        "held-out -ln p（训练后）": f"{post['eval_loss']:.4f}" if post else "未评测",
        "耗时(s)": round(runtime, 1) if runtime else "未记录",
    }
    safe_run("写报告", write_report, meta, before, after, rows)
    print("  报告：", RUN_DIR / "report.md")


if __name__ == "__main__":
    main()
