"""
阶段 5 · GRPOTrainer，外加「训练前 / 训练后」各测一次 held-out。

重点看三个参数：
  num_generations  每个 prompt 采几条（G，必须能整除有效 batch）
  beta             参考模型 KL 惩罚（默认 0 = 不挂参考模型，省一份显存）
  scale_rewards    "group" = 组内归一化，即 A_i = (r_i - mean) / (std + eps)
"""

from __future__ import annotations

from common import CKPT_DIR, hr
from grpo_rewards import accuracy_reward, format_reward


def train(rows, model_name: str, epochs: float, max_steps: int, num_gen: int,
          temperature: float, max_new: int, before_hook=None):
    hr("阶段 5 / 5   训练（每个 logging 步打印 loss / reward / 截断率）")
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    ds = Dataset.from_list([{"prompt": r["prompt"], "ground_truth": r["ground_truth"]} for r in rows])

    class LossPrinter(TrainerCallback):
        def __init__(self):
            self.rows = []

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
            self.rows.append(r)
            print(
                f"  step {r['step']:>3}  loss {r['loss']:.4f}  reward {r['reward']:.3f} "
                f"(std {r['reward_std']:.3f})  kl {r['kl']:.4f}  len {r['len']:.0f}  clip {r['clip']:.2f}"
            )

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
        report_to=[],
    )
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
    trainer.save_model(args.output_dir)

    post = {}
    if before_hook is not None:
        trainer.model.eval()
        post = before_hook(trainer.model)
        print(f"  训练后 held-out pass@1：{post['pass1']:.1%}   格式合规：{post['fmt']:.1%}")

    print()
    print(f"  训练结束：{result.metrics}")
    print(f"  权重已存：{args.output_dir}")
    return trainer, printer.rows, pre, post
