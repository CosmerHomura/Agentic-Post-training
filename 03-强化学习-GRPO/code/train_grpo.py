"""
第 3 步 · GRPO（Group Relative Policy Optimization）
没有教师、没有标准轨迹，只有"一个分数"。模型靠组内比较来自我提升。

运行：
    python train_grpo.py

数据：../data/grpo_prompts.jsonl —— prompt + ground_truth
"""

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from rewards import accuracy_reward, format_reward

dataset = load_dataset("json", data_files="../data/grpo_prompts.jsonl", split="train")

training_args = GRPOConfig(
    output_dir="./ckpt/grpo-math",

    # ---- 组采样：每个 prompt 采 G 条，组内比较 ----
    num_generations=8,

    # ---- 采样要"有随机性"，否则 8 条全一样，优势全为 0 ----
    temperature=1.0,

    # ---- 回答长度上限（单轮场景；多轮 agent 时这是整个 episode 的预算）----
    max_completion_length=512,

    # ---- 损失细节 ----
    loss_type="dapo",        # 默认；对超长回答更宽容（不按长度归一）
    scale_rewards="group",   # 默认；用组内标准差归一化优势
    beta=0.0,                # 对参考模型的 KL 惩罚；0 表示不惩罚（省一张模型的钱）

    # ---- 优化 ----
    learning_rate=1e-6,
    per_device_train_batch_size=8,   # 有效 batch 必须能被 num_generations 整除
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_steps=100,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[accuracy_reward, format_reward],  # 多个奖励会相加
)

trainer.train()
trainer.save_model(training_args.output_dir)

# 观察重点：
#   reward          —— 平均总奖励，应该稳步上升
#   reward_std      —— 组内标准差；长期接近 0 说明"全对"或"全错"，学不到东西
#   completions/mean_length —— 是否越写越长（reward hacking 的信号）
#   kl              —— beta=0 时恒为 0
