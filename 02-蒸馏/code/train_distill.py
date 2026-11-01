"""
第 2 步 · 蒸馏（On-policy GKD）—— 入口

和 SFT 最大的区别：**没有标准答案**。
数据里只有 prompt，答案由学生当场生成；教师逐 token 给出「每个词应该有多大概率」，
学生去对齐这个分布。

一条样本进模型前后：
    1) 只读 prompt              数据里没有 completion
    2) 套模板（add_generation_prompt=True）
    3) tokenize
    4) 学生自己生成 completion   on-policy：在自己写的东西上学
    5) 教师、学生对「同一个 next token」各给一个分布
       L = β·KL(p_T‖p_M) + (1-β)·KL(p_S‖p_M)，p_M = (1-β)p_S + β·p_T
    6) 训练、画损失、held-out 前后对比

用法（**在仓库根目录执行**）
--------------------------
    python 02-蒸馏/code/train_distill.py --inspect
    python 02-蒸馏/code/train_distill.py --explain-loss --beta 0.5
    python 02-蒸馏/code/train_distill.py
    python 02-蒸馏/code/train_distill.py --dataset trl-lib/ultrafeedback-prompt --limit 3000

代码结构（同目录 7 个文件）
--------------------------
    common.py           路径 / 打印 / tokenizer / 设备
    distill_data.py     阶段 1：读 prompt
    distill_prompt.py   阶段 2~3：模板与 tokenize
    distill_probe.py    阶段 4~5：学生生成、教师分布、held-out 探针
    distill_trainer.py  阶段 6：DistillationTrainer
    distill_report.py   ASCII 损失图 + report.md
    distill_loss.py     手算 KL/JSD（--explain-loss）
    train_distill.py    入口：解析参数、按顺序调用

产物（都在本章目录 02-蒸馏/ 下）
--------------------
    runs/distill/report.md     ← 跑完发我，我回填教程里的示例输出
    runs/distill/report.json
    runs/distill/loss.csv
    ckpt/distill-agent/
"""

from __future__ import annotations

import argparse

from common import safe_run, LOCAL_JSONL, STUDENT, TEACHER, print_env, RUN_DIR
from distill_data import load_prompts, stage_read
from distill_loss import explain_loss
from distill_probe import probe_kl, stage_distributions, stage_generate
from distill_prompt import stage_template, stage_tokens
from distill_report import write_report
from distill_trainer import train


def parse_args():
    ap = argparse.ArgumentParser(
        description="蒸馏全流程（on-policy GKD，带数据链路可视化与前后评测）",
        epilog="所有路径以仓库根目录为基准，命令在仓库根目录执行即可。",
    )
    ap.add_argument("--inspect", action="store_true", help="只跑数据链路（不需 GPU / 不需 trl）")
    ap.add_argument("--explain-loss", action="store_true", help="玩具分布手算 KL / JSD（不需 GPU）")
    ap.add_argument("--model", default=STUDENT, help="学生模型")
    ap.add_argument("--teacher", default=TEACHER, help="教师模型")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.5, help="0=前向KL 1=反向KL 0.5=JSD")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gen-tokens", type=int, default=160, help="学生生成 / 教师对比的长度上限")
    ap.add_argument("--tensorboard", action="store_true",
                    help="把日志写进 <章节>/runs/<x>/tb，供 TensorBoard / AutoPanel 查看")
    ap.add_argument("--plot-every", type=int, default=0, help="每多少步重画一次 loss.svg（默认 0 = 只在结束时画一次）")
    ap.add_argument("--data", default=str(LOCAL_JSONL))
    ap.add_argument("--dataset", default=None, help="HF 数据集名，如 trl-lib/ultrafeedback-prompt")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=2000, help="从官方数据集取前多少条 prompt")
    ap.add_argument("--eval-size", type=int, default=100, help="留出多少条做 held-out 评测")
    ap.add_argument("--no-eval", action="store_true", help="跳过 held-out 前后对比")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.explain_loss:
        explain_loss(args.beta)
        return

    print_env(args.model, args.teacher)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows, eval_rows, src = load_prompts(args)

    stage_read(rows, tok, src, len(eval_rows))
    stage_template(tok, rows)
    prompt_ids = stage_tokens(tok, rows)

    if args.inspect:
        print()
        print("  --inspect 模式结束（没有生成、没有训练）。")
        print("  去掉 --inspect 就会：学生生成 → 教师对比分布 → 训练 → 报告。")
        return

    student, gen = stage_generate(prompt_ids, tok, args.model, args.gen_tokens, args.temperature)
    dist_rows, mean_kl, loss, _ = stage_distributions(
        prompt_ids, gen, tok, student, args.model, args.teacher, args.beta, args.temperature
    )

    def hook(model, teacher):
        kl, samples = probe_kl(tok, model, teacher, eval_rows, min(args.gen_tokens, 120), args.temperature)
        return {"kl": kl, "samples": samples}

    _trainer, train_rows, pre, post = train(
        rows, args.model, args.teacher, args.epochs, args.max_steps,
        args.beta, args.temperature, args.gen_tokens, args.plot_every,
        tb=args.tensorboard,
        before_hook=None if args.no_eval or not eval_rows else hook,
    )
    print()
    print(f"  曲线图：{RUN_DIR / 'loss.svg'}")

    payload = {
        "student": args.model,
        "teacher": args.teacher,
        "beta": args.beta,
        "temperature": args.temperature,
        "gen_tokens": args.gen_tokens,
        "src": src,
        "n_samples": len(rows),
        "n_eval": len(eval_rows),
        "n_steps": len(train_rows),
        "mean_kl": mean_kl,
        "loss": loss,
        "dist_rows": dist_rows,
        "train_rows": train_rows,
        "pre": pre,
        "post": post,
    }
    safe_run("写报告", write_report, payload)
    print("  报告：", RUN_DIR / "report.md")


if __name__ == "__main__":
    main()
