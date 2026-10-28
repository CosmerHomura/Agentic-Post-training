"""
第 4 步 · 在环境里做 GRPO
与第 3 步的区别只有一处：把 reward_funcs 换成 environment_factory。
其余的训练循环、优势计算、损失，全都一样。

运行：
    python train_grpo_env.py

数据：本文件内联构造（也可以来自 jsonl，只要带 prompt 列）。
      每行除了 prompt，还会有 buggy_code / tests 两列 —— 它们会被 kwargs 传给 env.reset()。
"""

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from env_factory import CodingEnv

# ---------------------------------------------------------------- 1. 任务数据
rows = [
    {
        "prompt": "solution.py 里的测试失败了，请修好它。改完运行测试确认。",
        "buggy_code": "def divide(a, b):\n    return a / b\n",
        "tests": (
            "from solution import divide\n\n"
            "def test_ok():\n"
            "    assert divide(6, 3) == 2\n\n"
            "def test_zero():\n"
            "    try:\n"
            "        divide(1, 0)\n"
            "    except ValueError:\n"
            "        return\n"
            "    raise AssertionError('expected ValueError')\n"
        ),
    },
    {
        "prompt": "solution.py 缺少 multiply 函数，测试失败了，请补上。",
        "buggy_code": "def add(a, b):\n    return a + b\n",
        "tests": (
            "from solution import multiply\n\n"
            "def test_mul():\n"
            "    assert multiply(6, 7) == 42\n"
        ),
    },
]
dataset = Dataset.from_list(rows)

# ---------------------------------------------------------------- 2. 配置
training_args = GRPOConfig(
    output_dir="./ckpt/grpo-coding-env",

    # ★ 核心：环境取代奖励函数
    environment_factory=CodingEnv,   # 每次 rollout 新建一个实例

    num_generations=8,
    temperature=1.0,
    max_completion_length=4096,      # ★ 这是「整个 episode」的 token 预算，不是单轮
    max_tool_calling_iterations=20,  # 最多几轮工具调用

    loss_type="dapo",
    scale_rewards="group",
    beta=0.0,

    learning_rate=1e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    bf16=True,
    logging_steps=1,
    log_completions=True,            # 把完整 episode 打到日志里，调试环境必备
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
    # reward_funcs 留空：分数完全来自 env.get_reward()
    # 也可以给一个额外的函数，它会和 env 分数**相加**：
    # reward_funcs=[format_reward],
)

trainer.train()
trainer.save_model(training_args.output_dir)

# 观察重点：
#   reward                        —— 二元奖励，看通过率是否上升
#   tools/call_frequency          —— 平均几轮工具调用；太少=乱猜，太多=不会停
#   tools/failure_frequency       —— 工具失败率；应随训练下降
#   completions/clipped_ratio     —— 有多少 episode 撞到 token 预算上限
