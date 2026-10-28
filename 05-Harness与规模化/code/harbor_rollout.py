"""
第 5 步（二）· 黑盒 harness：让 mini-swe-agent / claude-code / opencode 来跑 rollout
这里"壳"不是你写的：真实 agent 在远程沙箱里跑完整循环，OpenEnv Harbor 把每一步的 token 抓下来。

前置：
  1) 起 Harbor 服务（它负责起沙箱、驱动 harness、记录 trace）：
       MAX_CONCURRENT_ENVS=9 openenv harbor serve \
         --dataset <你的任务数据集> --port 8200 --capture-port 8300 --expose gradio
  2) 起 vLLM，并允许 agent 从外部调用：
       vllm serve Qwen/Qwen3-4B \
         --enable-auto-tool-choice --tool-call-parser hermes \
         --logprobs-mode processed_logprobs \
         --return-tokens-as-token-ids \
         --weight-transfer-config '{"backend":"nccl"}'
  3) 沙箱凭证（例如 E2B_API_KEY）配好。
"""

from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.harness import HarnessRolloutWorker, HarborSessionFactory


# ---------------------------------------------------------------- 1. 奖励：从真实运行结果里算
# 入参是 HarnessRolloutOutcome，字段（按需取）：
#   .env_reward / .completion / .trace / .tool_call_count
#   .tool_failure_count / .tool_calls_by_name / .timed_out
# 返回 float；返回 None 表示"这条不可评分"，会被从组基线里剔除。
def rollout_reward_fn(outcome) -> float | None:
    if outcome.timed_out:
        return 0.0
    if outcome.env_reward is not None:
        return float(outcome.env_reward)

    # 没有环境分时，退化成"便宜的行为奖励"
    # 1) 跑完了并且真的动了工具
    if outcome.completion is None or outcome.tool_call_count == 0:
        return None
    score = 0.0
    # 2) 有最终答案
    if len(outcome.completion.split()) >= 5:
        score += 0.5
    # 3) 别把工具调崩
    if outcome.tool_failure_count == 0:
        score += 0.5
    return score


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    config = AsyncGRPOConfig(
        output_dir="./ckpt/harness-rl",
        vllm_server_url="http://localhost:8000",
        num_generations=8,
        max_inflight_tasks=8,        # 同时在跑的 agent 数（受 Harbor 并发上限约束）
        max_completion_length=8192,  # 真实 agent 的 episode 更长
        learning_rate=1e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1,
    )

    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-4B",
        args=config,
        # ★ 用 harness 代替自己写循环
        rollout_worker=HarnessRolloutWorker(
            # harness_session_factory：怎么连上"真实 agent"
            harness_session_factory=partial(
                HarborSessionFactory,
                "http://localhost:8200",      # Harbor 服务
                split="train",
                harness="mini-swe-agent",     # 或 "opencode" / "claude-code"
                sandbox="e2b",                # 远程沙箱后端
                llm_url="http://localhost:8000",   # agent 调用的推理服务（就是被训的模型）
                model="Qwen/Qwen3-4B",
            ),
            harness_adapter=None,          # 内置 harness 不用 adapter；自研 harness 时才要
            rollout_reward_fn=rollout_reward_fn,
            model_name="Qwen/Qwen3-4B",
            dataset=dataset,
            reward_funcs=[],
            processing_class=tokenizer,
            num_generations=8,
            max_inflight_tasks=8,
            vllm_server_url="http://localhost:8000",
        ),
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(config.output_dir)


# ---------------------------------------------------------------- 3. 自研 harness 怎么写 adapter
# 内置 harness 走 Harbor 的 TrainingTrace；自研 harness 只要把"同一段 token 序列"喂回来即可。
#
#   trace.node_id             这一轮的标识
#   trace.prompt_token_ids    这一轮看到的 token
#   trace.completion_token_ids 模型生成的 token
#   trace.per_token_logps     每个 token 的 logprob
#   trace.loss_mask           与 prompt+completion 等长，prompt 位置为 0
#
# 例：prompt [10, 11]、completion [20, 21, 22]、loss_mask [0, 0, 1, 0, 1]
#     → 只在 token 20 和 22 上学；21 被 mask 掉但仍作为上下文保留。
#     被整段 mask 的 completion 也不丢，它可以为后面的轮次提供上下文。

if __name__ == "__main__":
    main()
