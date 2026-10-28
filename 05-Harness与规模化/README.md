# 05 · Harness 与规模化

[← 上一节：RL 环境](../04-RL环境/README.md) · [总目录](../README.md) · [附录：资源与术语 →](../99-附录-资源与术语/README.md)

---

## 1. 一句话定义

**别自己写 agent 循环了——让真实 agent（mini-swe-agent / opencode / claude-code）在远程沙箱里跑，训练它。同时把"生成"和"训练"拆到两块 GPU 上并行，让 trainer 不再空转。**

这一步同时解决两件事：**真实性**（在它将来要跑的同一个壳里训练）和**速度**（长 episode 不再阻塞训练）。

---

## 2. 原文译文 · 5. Harnesses and scale

> 每一次 rollout 都是**一个完整的 agent 运行**——通过一个现成的 agent harness、在一个远程沙箱里跑完。用 agent 实际部署时会用的同一个 harness 来训练，意味着模型是在**它将来真正运行的那个环境**里学习。
>
> 生成与训练跑在**不同的 GPU** 上，所以很长的 episode 不会让训练器干等着。随后训练器把更新后的权重送回 vLLM。

### 原文推荐资源

| 类型 | 链接 |
|---|---|
| Slides | [Training a coding agent through a harness you did not write](https://huggingface.co/spaces/FineEnvs/multi-harness-rl-slides) |
| Read | [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) · [One sandbox per rollout](https://huggingface.co/blog/sergiopaniego/rl-environments-2026) · [Keep the Tokens Flowing: lessons from 16 open-source RL libraries](https://huggingface.co/blog/async-rl-training-landscape) |
| Docs | [Async GRPO Trainer](https://huggingface.co/docs/trl/async_grpo_trainer) · [OpenEnv harnesses](https://huggingface.co/docs/openenv/tutorials/harnesses) |

---

## 3. 为什么需要异步：同步 RL 的浪费

```
同步 GRPO 的时间线（一张卡干两件事）：

GPU: [生成 8 条 episode ............ 40s][训练一步 5s][生成 ............ 40s][训练 5s]
            ↑ 这 40 秒里，算损失的 GPU 完全闲置（利用率 ~11%）

异步（生成与训练分卡）：

GPU-0 (vLLM)  : [生成 episode 1][生成 2][生成 3][生成 4][生成 5]...  一直在产数据
GPU-1 (trainer):      [训练步1][训练步2][训练步3][训练步4][训练步5]...  一直在学
                        ↑ 队列里有货就吃，不吃就等（但不再空转）
```

一句话：**episode 越长，同步的浪费越致命**。第 4 步已经把 episode 拉长到几十轮，第 5 步拉到真实 agent 的几百轮——必须异步。

---

## 4. 完整架构

```
┌──────────────────────────── 训练侧（GPU-1）────────────────────────────┐
│  AsyncGRPOTrainer                                                      │
│    ① 从队列取 (prompt, completions, rewards, advantages, logprobs)     │
│    ② 前向 + 反向，算 clipped 策略梯度                                   │
│    ③ 每 weight_sync_steps 步：把新权重通过 NCCL 推给 vLLM               │
└───────────────▲───────────────────────────────────────┬────────────────┘
                │ 队列（进程间）                         │ 权重同步
┌───────────────┴───────────────────────────────────────▼────────────────┐
│  rollout worker（后台进程，CUDA_VISIBLE_DEVICES="" —— 不占 GPU）        │
│    ① 取 prompt → 发给 vLLM 生成                                  │
│    ② 打奖励分（reward_funcs / rollout_reward_fn）                       │
│    ③ 组内标准化算优势                                                   │
│    ④ 入队                                                               │
└───────────────▲────────────────────────────────────────────────────────┘
                │ HTTP
┌───────────────┴──────────────┐      ┌──────────────────────────────────┐
│  vLLM server（GPU-0）         │◀─────│  Harbor（远程沙箱编排）           │
│  · 生成 completion            │      │  · 每个 rollout 起一个沙箱        │
│  · 返回 token id + logprobs   │      │  · 跑 mini-swe-agent / opencode   │
└───────────────────────────────┘      │  · 抓 TrainingTrace 回传          │
                                       └──────────────────────────────────┘
```

### 关键设计：**同一个模型，两个身份**

vLLM 里跑的是"当前策略"，trainer 里更新的是同一份权重。同步靠 `weight_sync_steps`，而 `max_staleness` 决定"一条样本最多可以落后几次更新"——这是**异步带来的 off-policy 程度**的直接度量。

---

## 5. 核心数据结构：TrainingTrace

黑盒 harness 唯一必须交回来的东西，就是"这段 token 是我们模型生成的、那段不是"。

```
trace.node_id              = "turn_3"
trace.prompt_token_ids     = [10, 11]              ← 这一轮看到的上下文
trace.completion_token_ids = [20, 21, 22]          ← 模型生成的部分
trace.per_token_logps      = [-0.1, -0.9, -0.4]    ← 生成时的旧概率（算 ratio 用）
trace.loss_mask            = [0, 0, 1, 0, 1]       ← 与 prompt+completion 等长
```

```
位置:        0    1    2    3    4
token:      10   11   20   21   22
来源:     prompt prompt 模型 模型 模型
mask:       0    0    1    0    1
是否学:      ✗    ✗    ✓    ✗    ✓
```

**为什么 mask 里会有 0？** 因为 harness 可能在两轮之间插入了它自己生成的文本（比如它自己做了一次摘要、或某轮输出被判定无效）。这些位置**保留作为上下文**，但**不参与梯度**。

> 这就是"用别人的 harness 训练"的核心难点：你必须能从 harness 的日志里**精确还原**哪些 token 是模型生成的。
> Harbor 帮你做了这件事；自研 harness 就要自己写 adapter 产出 `TrainingTrace`。

---

## 6. 两条实现路径

| | 路径 A：自己写循环（第 4 步） | 路径 B：黑盒 harness（第 5 步） |
|---|---|---|
| 谁控制循环 | 你的 `environment_factory` | `mini-swe-agent` / `opencode` / `claude-code` |
| 工具从哪来 | 环境类的公开方法 | harness 自带的工具集 |
| 训练信号 | `env.get_reward()` | `rollout_reward_fn(outcome)` |
| token 对齐 | trainer 自己知道 | 必须靠 `TrainingTrace` 还原 |
| 适合 | 任务可控、想快速迭代 | 要"在真实部署配置里训练" |

### 路径 B 的执行流程

```
1. trainer 把 prompt 交给 rollout worker
2. worker 请求 Harbor：用哪个 harness、哪个沙箱、连哪个 vLLM
3. Harbor 起一个沙箱 → 启动 agent → agent 调用 vLLM（就是被训的模型）
4. agent 跑完（或超时）→ Harbor 收集 trace + 环境结果
5. worker 用 rollout_reward_fn 把 outcome 变成一个分数
6. 组内标准化 → 入队 → trainer 更新 → 权重同步回 vLLM
```

---

## 7. 输入输出

### 7.1 输入

| 输入 | 内容 |
|---|---|
| 任务数据集 | 至少 `prompt`；其余列可给 Harbor 做任务配置 |
| harness 配置 | `harness=`（哪个 agent）、`sandbox=`（哪种沙箱）、`llm_url=` |
| 奖励 | `rollout_reward_fn(outcome) -> float \| None`，或 `reward_funcs` |
| vLLM 服务 | 一个已启动、开了 `--weight-transfer-config` 的 server |

### 7.2 输出（`HarnessRolloutOutcome`）

| 字段 | 含义 |
|---|---|
| `completion` | agent 的最终回答（可能是 None） |
| `env_reward` | 环境/验证器给的分数（有就用它） |
| `trace` | `TrainingTrace` 列表，每轮一条 |
| `tool_call_count` | 总共调了几次工具 |
| `tool_failure_count` | 工具失败几次 |
| `tool_calls_by_name` | 每个工具被调了几次 |
| `timed_out` | 是否超时 |

> `rollout_reward_fn` 返回 `None` = **"这条不可评分"**，它会从组基线里被剔除（而不是当 0 分）。
> 这个细节很重要：把"跑挂了"当成 0 分会污染优势估计。

---

## 8. 代码与命令

- 异步训练：[`code/train_async_grpo.py`](code/train_async_grpo.py)
- 黑盒 harness：[`code/harbor_rollout.py`](code/harbor_rollout.py)

### 8.1 起服务（三个进程）

```bash
# ① Harbor：负责沙箱 + harness + trace 抓取
MAX_CONCURRENT_ENVS=9 openenv harbor serve \
  --dataset <hf-dataset> --port 8200 --capture-port 8300 --expose gradio

# ② vLLM：生成 + 权重接收
vllm serve Qwen/Qwen3-4B \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --logprobs-mode processed_logprobs \
  --return-tokens-as-token-ids \
  --weight-transfer-config '{"backend":"nccl"}'

# ③ 训练
CUDA_VISIBLE_DEVICES=1 accelerate launch train_async_grpo.py
```

### 8.2 训练脚本骨架

```python
trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen3-4B",
    args=AsyncGRPOConfig(
        vllm_server_url="http://localhost:8000",
        max_staleness=2,
        weight_sync_steps=8,
        num_generations=8,
        max_completion_length=8192,   # 真实 agent 的 episode 很长
    ),
    train_dataset=dataset,
    reward_funcs=[accuracy_reward],
)
```

### 8.3 黑盒 harness 骨架

```python
worker = HarnessRolloutWorker(
    harness_session_factory=partial(
        HarborSessionFactory, "http://localhost:8200",
        split="train", harness="mini-swe-agent", sandbox="e2b",
        llm_url="http://localhost:8000", model="Qwen/Qwen3-4B",
    ),
    harness_adapter=None,          # 内置 harness 不需要 adapter
    rollout_reward_fn=rollout_reward_fn,
    model_name="Qwen/Qwen3-4B",
    dataset=dataset,
    processing_class=tokenizer,
    num_generations=8,
    max_inflight_tasks=8,
    vllm_server_url="http://localhost:8000",
)
```

### 关键参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `vllm_server_url` | 生成服务地址 | 必须在另一块卡上 |
| `max_staleness` | 样本最多落后几次权重更新 | 1–3；越大越省但越 off-policy |
| `max_inflight_tasks` | 同时在跑的 rollout 数 | ≤ Harbor 的 `MAX_CONCURRENT_ENVS` |
| `weight_sync_steps` | 每几步同步权重到 vLLM | 4–16；太频繁会拖慢训练 |
| `max_completion_length` | episode 总预算 | 真实 agent 建议 8k–32k |
| `ignore_data_skip` | 续训时跳过已训数据 | 默认 True |

---

## 9. 产物与验收

### 9.1 指标

| 指标 | 含义 | 健康表现 |
|---|---|---|
| `reward` | 平均奖励 | 上升 |
| `reward_std` | 组内差异 | 0.1–0.5 |
| `tools/call_frequency` | agent 平均工具调用数 | 应稳定在某个值，不无限涨 |
| `tools/failure_frequency` | 工具失败率 | 下降 |
| `completions/mean_length` | episode 长度 | 稳定；暴涨=兜圈子 |
| `completions/clipped_ratio` | 撞预算比例 | < 10% |
| `kl` | 与参考模型偏离 | 平缓 |

**异步特有的观察项**：

```
□ 训练 loss 是否比同步版本更抖？（正常，因为样本来自稍旧的策略）
□ max_staleness 调大后 reward 是否还涨？（不涨说明 off-policy 太重）
□ 权重同步是否成为瓶颈？（看 step_time 是否周期性飙高）
```

### 9.2 最终验收：在真实 harness 里跑一次

训练结束后，最好的验收不是看 loss，而是：

```
□ 用同一个 harness（mini-swe-agent）在留出任务上跑 20 个 episode
□ 统计通过率，对比训练前
□ 抽 3 条 transcript 人工看：工具调用是否合理、是否读报错、是否收敛
```

### 9.3 产物

- `./ckpt/harness-rl/`：在真实 agent 壳里训练过的模型
- `rollout_state.json`：断点续训用的状态（首个未训练 prompt 的位置）
- 一套可复现的服务编排（Harbor + vLLM + 沙箱）

---

## 10. 常见坑

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | 用了 lambda / 闭包做奖励 | `TypeError: can't pickle` | 改成模块级函数或 `functools.partial` |
| 2 | 工具写成嵌套函数 | pickle 失败 / 工具没注册 | 工具必须模块级，靠 `__name__` 注册 |
| 3 | 在 rollout 进程里加载奖励模型 | 报无 GPU 可用 | rollout 进程 `CUDA_VISIBLE_DEVICES=""`，奖励模型另起 HTTP 服务 |
| 4 | Harbor 并发 < `max_inflight_tasks` | rollout 排队，GPU 饿死 | 提高 `MAX_CONCURRENT_ENVS` |
| 5 | 忘记 `--weight-transfer-config` | 权重同步失败 | 两台机器都要配 NCCL 后端 |
| 6 | vLLM 与 trainer 同卡 | OOM | 必须分卡（第 5 步的前提） |
| 7 | `max_staleness` 设太大 | reward 不涨 | 降到 1–2 |
| 8 | trace 的 mask 造错 | 模型学到"harness 说的话" | 用 Harbor 的 trace；自研 harness 要严格核对 |
| 9 | 沙箱超时没设 | 挂死的 episode 拖垮队列 | 用 `timed_out` 字段识别并返回 0 或 None |
| 10 | 按次计费的沙箱 | 账单爆炸 | 控制 `max_inflight_tasks` 与 episode 上限 |

---

## 11. 练习与自测

1. **异步 vs 同步**：同一任务跑 200 步同步 GRPO 与异步 GRPO，比较墙钟时间与最终奖励。算一下"理论加速比 = 生成时间/训练时间"。
2. **staleness 实验**：`max_staleness` 取 1 / 4 / 8，观察 `reward` 曲线。你会看到"省时间"与"学得准"的权衡。
3. **换 harness**：把 `harness="mini-swe-agent"` 换成 `"opencode"`，同一份权重直接跑，观察行为差异——这直观说明"壳"对模型表现的影响有多大。

---

## 12. 五步总复习：什么时候用哪一步

```
你有「正确的完整轨迹」吗？
 ├─ 有 ──────────────────────────────▶ ① SFT（最便宜、最稳，先做这个）
 └─ 没有
     ├─ 有一个更强的教师模型吗？
     │    ├─ 有 ──────────────────────▶ ② 蒸馏（信号密、样本省）
     │    └─ 没有
     │        ├─ 能写出「判断对错」的规则吗？
     │        │    ├─ 能，且一轮就够 ──▶ ③ GRPO + 奖励函数
     │        │    └─ 能，但要动手试 ──▶ ④ GRPO + 环境
     │        └─ 只能事后看结果
     │             └─ 有真实 agent 壳吗？
     │                  └─ 有 ────────▶ ⑤ 异步 GRPO + Harness
     └─ 什么都没有 ──▶ 回去先造数据 / 造验证器
```

**真实流水线通常是串起来的**：`SFT（学会格式）→ 蒸馏或 GRPO（学会解题）→ 环境 / Harness（学会在真实壳里行动）`。

---

## 13. 一句话带走

> 前四步你在**造数据、造奖励、造环境**；第五步你终于**不再造任何东西**——直接用生产环境里的那个壳。
> 代价是工程复杂度：三个进程、两块卡、pickle 约束、权重同步、沙箱账单。
> 这也是为什么原文把"规模化"和"harness"放在同一步：**到了这一步，瓶颈已经不是算法，而是流水线工程。**

---

[← 上一节：RL 环境](../04-RL环境/README.md) · [总目录](../README.md) · [附录：资源与术语 →](../99-附录-资源与术语/README.md)
