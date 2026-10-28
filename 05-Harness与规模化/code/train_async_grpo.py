"""
第 5 步（一）· 异步 GRPO：把"生成"和"训练"拆到两块 GPU 上
算法没变（还是 GRPO），变的是流水线：
  · rollout worker 在后台不停地采样、打分、算优势、入队
  · 训练循环只管从队列取样本、算损失、更新权重
  · 每 weight_sync_steps 步，用 NCCL 把新权重推回 vLLM

启动方式（两台终端，两块卡）：
  终端 1（生成）：
    CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
      vllm serve Qwen/Qwen3-4B --max-model-len 4096 \
      --logprobs-mode processed_logprobs \
      --weight-transfer-config '{"backend":"nccl"}'

  终端 2（训练）：
    CUDA_VISIBLE_DEVICES=1 accelerate launch train_async_grpo.py
"""

from datasets import load_dataset
from trl import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.rewards import accuracy_reward      # ★ 必须是模块级函数（要 pickle）

def main():
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    training_args = AsyncGRPOConfig(
        output_dir="./ckpt/async-grpo",

        # ---- 指向你启动的 vLLM 服务 ----
        vllm_server_url="http://localhost:8000",

        # ---- 异步流水线的三个关键旋钮 ----
        max_staleness=2,          # 一条样本最多可以落后几次权重更新
        # max_inflight_tasks 默认 = max_staleness × per_device_bs × grad_accum × 进程数
        weight_sync_steps=8,      # 每 8 步把权重同步回 vLLM

        # ---- 与同步 GRPO 相同的部分 ----
        num_generations=8,
        temperature=1.0,
        max_completion_length=2048,
        learning_rate=1e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        bf16=True,
        logging_steps=1,
        ignore_data_skip=True,    # 断点续训时跳过已训数据（异步默认行为）
    )

    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-4B",
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[accuracy_reward],
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()

# ★ 三个必读的约束（踩了就一定报错）：
#   1. reward_funcs / tools / environment_factory 必须可 pickle：
#      → 模块级函数、functools.partial、可调用类的实例；lambda 和闭包会在 train() 抛 TypeError
#   2. 工具按 __name__ 注册，所以工具函数也必须定义在模块顶层
#   3. rollout 进程的 CUDA_VISIBLE_DEVICES=""（不占 GPU），
#      所以奖励模型不能塞在 rollout 进程里，要另起一个推理服务用 HTTP 调
