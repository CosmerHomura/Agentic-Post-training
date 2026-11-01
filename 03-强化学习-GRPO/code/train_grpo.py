"""
第 3 步 · 强化学习（GRPO）—— 入口

数据里只有 prompt 和「怎么判对错」，没有答案轨迹。
模型自己写 G 条，奖励函数逐条打分，组内比较算出「谁比平均水平好」，
然后提高好答案的似然、压低差答案的似然。

一条样本进训练，发生这些事：
    1) 读数据          prompt + ground_truth（判分用），没有 completion
    2) 套模板 + tokenize
    3) 采 G 条 completion（同一个 prompt，独立采样 8 次）
    4) 逐条打分        reward = format + accuracy
    5) 组内算优势      A_i = (r_i - mean) / (std + eps)
    6) 策略梯度损失     -min(ratio·A, clip(ratio,1±ε)·A)，只在 completion 段算
    7) 训练、奖励曲线、held-out 前后对比

用法（**在仓库根目录执行**）
--------------------------
    python 03-强化学习-GRPO/code/train_grpo.py --inspect
    python 03-强化学习-GRPO/code/train_grpo.py --explain-loss
    python 03-强化学习-GRPO/code/train_grpo.py --max-steps 30 --num-generations 4
    python 03-强化学习-GRPO/code/train_grpo.py --dataset trl-lib/DeepMath-103K --limit 2000

代码结构（同目录 8 个文件）
--------------------------
    common.py          路径 / 打印 / tokenizer 小工具 / 环境
    grpo_data.py       阶段 1：读数据、答案归一化、格式正则
    grpo_rewards.py    奖励函数（格式分 / 正确性分 / 组内优势）
    grpo_rollout.py    阶段 2~3：模板、采样打分、held-out pass@1
    grpo_trainer.py    阶段 5：GRPOTrainer
    grpo_report.py     ASCII 曲线 + report.md
    grpo_loss.py       手算 GRPO 损失（--explain-loss）
    train_grpo.py      入口：解析参数、按顺序调用

产物（都在本章目录 03-强化学习-GRPO/ 下）
--------------------
    runs/grpo/report.md       ← 跑完发我，我回填教程里的示例输出
    runs/grpo/report.json
    ckpt/grpo-agent/
"""

from __future__ import annotations

import argparse

from common import LOCAL_JSONL, MODEL_DEFAULT, print_env
from grpo_data import load_rows, stage_read
from grpo_loss import explain_loss
from grpo_report import write_report
from grpo_rollout import pass_at_1, stage_sample_and_score, stage_template
from grpo_trainer import train


def parse_args():
    ap = argparse.ArgumentParser(
        description="GRPO 全流程（带数据链路可视化、奖励打印、held-out 前后评测）",
        epilog="所有路径以仓库根目录为基准，命令在仓库根目录执行即可。",
    )
    ap.add_argument("--inspect", action="store_true", help="只跑数据链路（不需 GPU）")
    ap.add_argument("--explain-loss", action="store_true", help="玩具数字手算 GRPO 损失（不需 GPU）")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--num-generations", type=int, default=8, help="G：每个 prompt 采几条")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=256, help="单条 completion 的 token 上限")
    ap.add_argument("--data", default=str(LOCAL_JSONL))
    ap.add_argument("--dataset", default=None, help="HF 数据集名，如 trl-lib/DeepMath-103K")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--eval-size", type=int, default=100, help="留出多少条做 held-out")
    ap.add_argument("--tensorboard", action="store_true",
                    help="把日志写进 <章节>/runs/<x>/tb，供 TensorBoard / AutoPanel 查看")
    ap.add_argument("--plot-every", type=int, default=20, help="每多少步重画一次 runs/grpo/loss.svg（0=不画）")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.explain_loss:
        explain_loss()
        return

    print_env(args.model)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows, eval_rows, src = load_rows(args)

    stage_read(rows, tok, src, len(eval_rows))
    stage_template(rows, tok)

    if args.inspect:
        print()
        print("  --inspect 模式结束（没有采样、没有训练）。")
        print("  去掉 --inspect 就会：采 G 条 → 打分 → 算优势 → 训练 → 报告。")
        return

    stats, comps = stage_sample_and_score(
        rows, tok, args.model, args.num_generations, args.temperature, args.max_new
    )

    def hook(model):
        return pass_at_1(tok, model, eval_rows, min(args.max_new, 256))

    _trainer, train_rows, pre, post = train(
        rows, args.model, args.epochs, args.max_steps,
        args.num_generations, args.temperature, args.max_new, args.plot_every,
        tb=args.tensorboard,
        before_hook=hook if eval_rows else None,
    )

    print()
    print(f"  曲线图：{RUN_DIR / 'loss.svg'}")

    write_report(
        {
            "model": args.model,
            "num_gen": args.num_generations,
            "temperature": args.temperature,
            "max_new": args.max_new,
            "n_samples": len(rows),
            "n_eval": len(eval_rows),
            "src": src,
            "n_steps": len(train_rows),
            **{k: stats[k] for k in ("f_scores", "a_scores", "mean", "std", "adv")},
            "train_rows": train_rows,
            "pre": pre,
            "post": post,
        }
    )


if __name__ == "__main__":
    main()
