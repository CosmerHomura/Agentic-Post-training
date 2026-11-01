"""
阶段 5 · GRPOTrainer，外加「训练前 / 训练后」各测一次 held-out。

重点看三个参数：
  num_generations  每个 prompt 采几条（G，必须能整除有效 batch）
  beta             参考模型 KL 惩罚（默认 0 = 不挂参考模型，省一份显存）
  scale_rewards    "group" = 组内归一化，即 A_i = (r_i - mean) / (std + eps)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

from common import (LOG_EVERY, safe_save, CKPT_DIR, RUN_DIR, csv_append,
                    fmt_eta, hr, plot_svg, spark)
from grpo_rewards import accuracy_reward, format_reward


def train(rows, model_name: str, epochs: float, max_steps: int, num_gen: int,
          temperature: float, max_new: int, plot_every: int = 20,
          before_hook=None, tb: bool = False):
    hr("阶段 5 / 5   训练（每个 logging 步打印 loss / reward / 截断率）")
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    ds = Dataset.from_list([{"prompt": r["prompt"], "ground_truth": r["ground_truth"]} for r in rows])

    class LossPrinter(TrainerCallback):
        """每步一行实时进度；每 plot_every 步把两条曲线重写成 runs/grpo/loss.svg。"""

        def __init__(self):
            self.rows = []
            self.csv = RUN_DIR / "loss.csv"
            try:
                if self.csv.exists():
                    self.csv.unlink()      # 新一轮从零开始，别把上次的接在后面
            except Exception:
                pass
            self.ema = None
            self.t0 = time.time()
            self.tty = sys.stdout.isatty()
            self.tb = None          # TensorBoardCallback，训练器建好后 bind() 进来

        def bind(self, trainer):
            """找到 TensorBoard 回调，好在同一根事件流里多写一条平滑曲线。"""
            try:
                for cb in trainer.callback_handler.callbacks:
                    if type(cb).__name__ == "TensorBoardCallback":
                        self.tb = cb
                        return
            except Exception:
                self.tb = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            r = {
                "step": state.global_step,
                "loss": round(float(logs["loss"]), 4),
                "reward": round(float(logs.get("reward", 0) or 0), 4),
                "reward_std": round(float(logs.get("reward_std", 0) or 0), 4),
                "kl": round(float(logs.get("kl", 0) or 0), 4),
                "len": round(float(logs.get("completions/mean_length", 0) or 0), 1),
                "clip": round(float(logs.get("completions/clipped_ratio", 0) or 0), 3),
            }
            csv_append(self.csv, r)     # 先落盘：后面崩了也留得住
            self.rows.append(r)
            self.ema = r["loss"] if self.ema is None else 0.9 * self.ema + 0.1 * r["loss"]
            if self.tb is not None and getattr(self.tb, "tb_writer", None) is not None:
                try:
                    self.tb.tb_writer.add_scalar("train/loss_ema", self.ema, state.global_step)
                except Exception:
                    pass
            total = state.max_steps or 0
            if LOG_EVERY > 1 and r["step"] % LOG_EVERY and r["step"] != total:
                return
            pct = 100.0 * r["step"] / total if total else 0.0
            eta = ""
            if total and r["step"]:
                eta = "  eta " + fmt_eta((time.time() - self.t0) / r["step"] * (total - r["step"]))
            line = (f"step {r['step']:>4}/{total:<4} {pct:5.1f}%  loss {r['loss']:.4f}  "
                    f"reward {r['reward']:.3f}±{r['reward_std']:.3f}  len {r['len']:.0f}{eta}  "
                    f"{spark([x['reward'] for x in self.rows])}")
            if self.tty:
                print("\r\033[K" + line[:160], end="", flush=True)
            else:
                print("  " + line, flush=True)
            if plot_every and r["step"] % plot_every == 0:
                self._plot(total)

        def _plot(self, total):
            steps = [x["step"] for x in self.rows]
            losses = [x["loss"] for x in self.rows]
            rewards = [x["reward"] for x in self.rows]
            ema, e = [], None
            for v in losses:
                e = v if e is None else 0.9 * e + 0.1 * v
                ema.append(e)
            plot_svg(
                RUN_DIR / "loss.svg",
                [
                    {"name": "loss（EMA 0.9）", "pts": list(zip(steps, ema)), "color": "#2563eb", "width": 2.0},
                    {"name": "平均 reward", "pts": list(zip(steps, rewards)), "color": "#16a34a", "width": 1.8},
                ],
                title=f"GRPO 曲线（{steps[-1]}/{total or steps[-1]} 步）", ylabel="loss / reward",
            )

        def on_train_end(self, args, state, control, **kwargs):
            if self.tty:
                print()
            if self.rows:
                self._plot(state.max_steps or self.rows[-1]["step"])

    printer = LossPrinter()
    has_cuda = torch.cuda.is_available()
    args = GRPOConfig(
        output_dir=str(CKPT_DIR),
        model_init_kwargs={"dtype": "bfloat16" if has_cuda else "float32"},
        use_cpu=not has_cuda,
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",
        num_generations=num_gen,          # ★ G：每个 prompt 采几条
        temperature=temperature,          # ★ 采样温度
        max_completion_length=max_new,    # ★ 单条答案的 token 上限
        beta=0.0,                         # ★ 0 = 不挂参考模型 KL（省一份显存）
        loss_type="dapo",                 # ★ TRL 1.16 默认
        scale_rewards="group",            # ★ 组内归一化 → 就是 A_i 那个公式
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps else -1,
        per_device_train_batch_size=num_gen,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        warmup_steps=3,
        lr_scheduler_type="cosine",
        bf16=has_cuda,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,      # reward 要用数据集里的 ground_truth 列
        report_to=["tensorboard"] if tb else [],
        run_name=RUN_DIR.name,
    )
    # 两个都设，因为不同版本读的地方不一样：
    #   transformers 5.x 的 TensorBoardCallback 只认环境变量 TENSORBOARD_LOGGING_DIR，
    #   4.x 才读 args.logging_dir；TRL 1.16 的 *Config.__init__ 又不接受 logging_dir 入参。
    # 不设的话，5.x 会退到 os.path.join(args.output_dir, "runs/<日期>_<主机名>")，很能跑偏。
    tb_dir = str(RUN_DIR / "tb")
    args.logging_dir = tb_dir
    if tb:
        os.environ["TENSORBOARD_LOGGING_DIR"] = tb_dir
    if tb and importlib.util.find_spec("tensorboard") is None:
        print('  ⚠ 没装 tensorboard，日志写不进去：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard')

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[format_reward, accuracy_reward],
        args=args,
        train_dataset=ds,
        callbacks=[printer],
    )
    print(f"  模型参数：{sum(p.numel() for p in trainer.model.parameters()) / 1e6:.0f}M")
    print(f"  有效 batch：{num_gen} 条/prompt × 2 步累积")
    print()

    pre = {}
    if before_hook is not None:
        trainer.model.eval()
        pre = before_hook(trainer.model)
        print(f"  训练前 held-out pass@1：{pre['pass1']:.1%}   格式合规：{pre['fmt']:.1%}")

    print()
    result = trainer.train()
    safe_save(trainer, args.output_dir)   # 先落盘，后面的评估/报告挂掉也不丢

    post = {}
    if before_hook is not None:
        try:
            trainer.model.eval()
            post = before_hook(trainer.model)
            print(f"  训练后 held-out pass@1：{post['pass1']:.1%}   格式合规：{post['fmt']:.1%}")
        except Exception as e:      # noqa: BLE001
            print(f"  ⚠ 训练后评估失败（权重已保存，不受影响）：{type(e).__name__}: {e}")

    print()
    print(f"  训练结束：{result.metrics}")
    return trainer, printer.rows, pre, post
