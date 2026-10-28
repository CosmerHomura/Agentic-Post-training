"""
第 1 步 · 监督微调（SFT）
用一个「强模型 / 人类修好 bug」的完整轨迹，教小模型学会同样的行为。

运行：
    python train_sft.py

数据：../data/sft_agent_trace.jsonl
      每行 {"tools": [...], "messages": [...]}，assistant 的 tool_calls 与 tool 结果都在 messages 里。
"""

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------- 1. 数据
# 对话式格式：TRL 会自动套 tokenizer 的 chat template，
# 并把 assistant 的 tool_calls 渲染成 <tool_call>...</tool_call>（取决于模板）。
dataset = load_dataset("json", data_files="../data/sft_agent_trace.jsonl", split="train")

# ---------------------------------------------------------------- 2. 配置
training_args = SFTConfig(
    output_dir="./ckpt/sft-agent",
    model_init_kwargs={"dtype": "bfloat16"},

    # --- 关键：只对 assistant 的 token 算 loss ---
    # 需要 chat template 里带 {% generation %} 标记；Qwen3 等已知家族 TRL 会自动打补丁。
    assistant_only_loss=True,

    # --- 吞吐 ---
    packing=True,              # 把多条样本拼进同一序列，省 padding；短样本多时收益大
    max_length=4096,           # 一条 agent 轨迹可能很长，别设太小

    # --- 优化 ---
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,        # SFT 常用 1e-5 ~ 2e-5；LoRA 可到 1e-4
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_strategy="epoch",
)

# ---------------------------------------------------------------- 3. 训练
trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=training_args,
    train_dataset=dataset,
    # processing_class 不传时按 model 自动取名；换 tokenizer 时显式传
)

trainer.train()
trainer.save_model(training_args.output_dir)

# ---------------------------------------------------------------- 4. 自检
# 训练后应该看到 loss 稳定下降、mean_token_accuracy 上升；
# 然后手动跑一次：模型是否学会了「先 read_file 再改」的顺序，而不是直接瞎写。
