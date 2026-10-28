"""
第 2 步 · 蒸馏（On-policy GKD）
学生自己写，教师逐 token 给"每个词应该有多大概率"，学生去对齐这个分布。

运行：
    python train_distill.py

数据：../data/distill_prompts.jsonl —— 只有 prompt，没有答案。
      答案由"学生模型当场生成"，这正是 on-policy 的含义。
"""

from datasets import load_dataset
from trl import DistillationConfig, DistillationTrainer

dataset = load_dataset("json", data_files="../data/distill_prompts.jsonl", split="train")

training_args = DistillationConfig(
    output_dir="./ckpt/distill-agent",

    # ★ 核心：β 选择用哪种散度去对齐师生分布
    #   β = 0.0  → 前向 KL  D_KL[p_teacher ‖ p_student]  学生覆盖教师所有模式，更"散"
    #   β = 1.0  → 反向 KL  D_KL[p_student ‖ p_teacher]  学生只学教师最确定的模式，更"尖"（默认值）
    #   β = 0.5  → 标准 JSD
    beta=0.5,

    # ★ 温度：同时作用于"生成"和"损失"。T 越大，教师分布越软，信息越多
    temperature=1.0,

    # 学生一次最多生成多少 token（多轮工具调用时这是总预算）
    max_completion_length=1024,

    # 优化
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,          # 蒸馏默认 lr 比 SFT 小
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,

    # 想快就去掉下面两行注释，用 vLLM 生成学生的 rollout
    # use_vllm=True,
    # vllm_mode="colocate",
)

trainer = DistillationTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",        # 学生
    teacher_model="Qwen/Qwen2.5-1.5B-Instruct",  # 教师（更大/更强）
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(training_args.output_dir)

# 注意：
# 1) 学生与教师必须共用同一个 tokenizer（否则 token 下标对不上）。
# 2) 每一步训练都要额外跑一次教师前向 → 比 SFT 慢，显存也要多留一份。
# 3) 这里没有"标准答案"，只有"教师认为什么更可能"。
