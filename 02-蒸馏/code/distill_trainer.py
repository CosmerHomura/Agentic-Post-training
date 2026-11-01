"""
阶段 6：交给 TRL 的 DistillationTrainer 训练。

和 SFT 的差别只有三处，但每一处都会让「训练前 vs 训练后」的数字讲同一个故事：
  · 数据只有 prompt，completion 由学生现场生成（on-policy）
  · 损失是分布距离（β 混合 KL/JSD），不是 -ln p
  · 评测指标是「held-out 上学生与教师的平均前向 KL」
"""

from __future__ import annotations

from common import CKPT_DIR, hr


def train(rows, student_name, teacher_name, epochs: float, max_steps: int | None,
          beta: float, temperature: float, gen_tokens: int, before_hook=None):
    hr("阶段 6   训练（学生每个 logging 步打印一次 loss）")
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import DistillationConfig, DistillationTrainer

    ds = Dataset.from_list([{"prompt": r["prompt"]} for r in rows])

    class LossPrinter(TrainerCallback):
        def __init__(self):
            self.rows = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            self.rows.append(
                {
                    "step": state.global_step,
                    "loss": round(float(logs["loss"]), 4),
                    "lr": float(logs.get("learning_rate", 0.0)),
                    "grad_norm": round(float(logs.get("grad_norm", 0) or 0), 4),
                    "entropy": round(float(logs.get("entropy", 0) or 0), 4),
                    "mean_len": round(float(logs.get("completions/mean_length", 0) or 0), 1),
                    "clip_ratio": round(float(logs.get("completions/clipped_ratio", 0) or 0), 3),
                }
            )
            r = self.rows[-1]
            print(
                f"  step {r['step']:>3}  loss {r['loss']:.4f}  lr {r['lr']:.2e}  "
                f"grad {r['grad_norm']:.3f}  ent {r['entropy']:.2f}  "
                f"len {r['mean_len']:.0f}  clip {r['clip_ratio']:.2f}"
            )

    printer = LossPrinter()
    has_cuda = torch.cuda.is_available()
    args = DistillationConfig(
        output_dir=str(CKPT_DIR),
        model_init_kwargs={"dtype": "bfloat16" if has_cuda else "float32"},
        use_cpu=not has_cuda,
        optim="adamw_torch_fused" if has_cuda else "adamw_torch",
        beta=beta,                      # ★ 用哪种散度
        temperature=temperature,        # ★ 同时作用于生成与损失
        max_completion_length=gen_tokens,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps else -1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,             # ★ 蒸馏默认比 SFT 小
        warmup_steps=3,
        lr_scheduler_type="cosine",
        bf16=has_cuda,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = DistillationTrainer(
        model=student_name,
        teacher_model=teacher_name,
        args=args,
        train_dataset=ds,
        callbacks=[printer],
    )
    print(f"  学生参数：{sum(p.numel() for p in trainer.model.parameters()) / 1e6:.0f}M")
    print(f"  教师参数：{sum(p.numel() for p in trainer.teacher_model.parameters()) / 1e6:.0f}M")
    print()

    # ---- 训练前：先在 held-out 上量一次「离教师多远」（必须在 train 之前）----
    pre = {}
    if before_hook is not None:
        trainer.model.eval()
        pre = before_hook(trainer.model, trainer.teacher_model)
        print(f"  训练前 held-out 平均 KL：{pre['kl']:.4f}")

    print()
    result = trainer.train()
    trainer.save_model(args.output_dir)

    post = {}
    if before_hook is not None:
        trainer.model.eval()
        post = before_hook(trainer.model, trainer.teacher_model)
        print(f"  训练后 held-out 平均 KL：{post['kl']:.4f}")

    print()
    print(f"  训练结束：{result.metrics}")
    print(f"  权重已存：{args.output_dir}")
    return trainer, printer.rows, pre, post
