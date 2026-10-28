# 03 · 强化学习（GRPO）

[← 上一节：蒸馏](../02-蒸馏/README.md) · [总目录](../README.md) · [下一节：RL 环境 →](../04-RL环境/README.md)

---

## 1. 一句话定义

**让模型对同一个问题写 8 个答案，打分，然后"把比平均分高的那些答案变得更可能"。**

不需要教师，不需要标准轨迹 —— 只需要一个**能判断对错的东西**（奖励函数）。这就是从"抄"转向"练"的分水岭。

---

## 2. 原文译文 · 3. Reinforcement learning (RL)

> 在强化学习中，模型从**对自己输出的一个分数**里学习。它写一个答案，奖励函数给它打分，训练让高分答案变得更可能。这样就**不再需要教师**——只要能校验答案就够了。
>
> 这一步用的算法是 **GRPO（Group Relative Policy Optimization）**。对每个 prompt，它采样若干个答案，给它们全部打分，然后提高那些**得分高于组内平均**的答案的概率。在组内比较答案，就告诉了模型哪些尝试更好，**而不需要另外训练一个模型来估计这件事**（这正是它相对 PPO 省掉 critic 的地方）。
>
> 这个信号比蒸馏弱：**整条答案只有一分数**，而不是每个 token 一个值，所以模型需要很多次尝试才能学到东西。用普通奖励函数时，每个答案只有一轮、只打一次分。而**环境**让模型可以行动多次，并看到每个动作的结果。

### 原文推荐资源

| 类型 | 链接 |
|---|---|
| Watch | [Training Agents 3: reinforcement learning](https://www.youtube.com/watch?v=ztdTed5egrM)（[合集](https://huggingface.co/collections/sergiopaniego/training-agents-class-3-reinforcement-learning-6ad6aa71050342629dd0e3ec)） |
| Read | [How frontier models train on outcomes in 2026](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) |
| Project | [Fine-tuning a 350M model for structured outputs in 100 GRPO steps](https://huggingface.co/blog/grpo-with-trl-ifstruct) |
| Docs | [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) |

---

## 3. GRPO 到底在算什么

### 3.1 四个步骤（TRL 文档的原话顺序）

```
① 生成       每个 prompt 采样 G 条 completion（G = num_generations，默认 8）
② 算优势     用同一 prompt 的 G 个奖励做组内标准化
③ 估 KL      与参考模型的 KL（beta>0 时才需要参考模型）
④ 算损失     对 completion 的每个 token 做 clipped 策略梯度
```

### 3.2 优势：为什么"组内比较"就够

```
prompt: "23 × 47 = ?"
   ├── 回答1  <answer>1081</answer>   reward 1.0   ┐
   ├── 回答2  <answer>1081</answer>   reward 1.0   │ 组内均值 μ = 0.5
   ├── 回答3  <answer>1000</answer>   reward 0.0   │ 组内标准差 σ = 0.5
   ├── 回答4  <answer>1081</answer>   reward 1.0   ┘
   ...
                    r_i - μ
        A_i = ───────────────────
                     σ + ε

   → 答对的 A ≈ +1，答错的 A ≈ -1
   → 训练目标：让 A>0 的整条序列概率上升，A<0 的下降
```

**关键洞察**：因为基线 μ 来自"同一 prompt 的其他尝试"，所以**不需要 critic 网络**去估计"这个状态值多少"。这就是 GRPO 比 PPO 省一半显存的原因。

### 3.3 损失（每个 completion token 上）

```
ratio_t = π_θ(y_t | y_<t) / π_old(y_t | y_<t)

L = -E[ min( ratio_t · A,  clip(ratio_t, 1-ε, 1+ε) · A ) ]
```

`clip` 是防止一步更新太猛把策略带崩。`loss_type="dapo"`（默认）在此之上不按回答长度归一，避免长回答被系统性压低。

---

## 4. 训练信号从哪来

| 环节 | 内容 |
|---|---|
| **输入** | `prompt`（+ 数据集里的额外列，如 `ground_truth`） |
| **模型产出** | 每个 prompt **G 条** completion |
| **监督信号** | 每条 completion 一个标量 reward（可以是多个奖励相加） |
| **损失** | 组内标准化优势 × clipped 策略梯度 |
| **产物** | 会"拿高分"的模型——前提是你的奖励函数真的量对了东西 |

```
        prompt ──┬──▶ completion 1 ──▶ reward 1.0 ┐
                 ├──▶ completion 2 ──▶ reward 0.0 │
                 ├──▶ completion 3 ──▶ reward 1.0 │──▶ 组内标准化 ──▶ A_i ──▶ loss
                 └──▶ ...             ──▶ reward 0.0 ┘
```

---

## 5. 输入格式

### 5.1 数据集：prompt + 你想要的任何额外列

`data/grpo_prompts.jsonl`：

```json
{"prompt": [{"role": "user", "content": "计算 23 × 47 的值。……写进 <answer></answer>。"}],
 "ground_truth": "1081"}
```

| 列 | 必需 | 说明 |
|---|---|---|
| `prompt` | ✅ | 对话式 messages（或字符串，TRL 会套模板） |
| `ground_truth` | 通常需要 | **自定义列名**，会作为 kwargs 传给奖励函数 |
| `tools` | 可选 | 工具 schema，多轮工具调用时需要 |
| `environment` | 可选 | 多环境时用：填环境名，路由到对应的 factory（见第 4 章） |
| 其他任意列 | 可选 | 全部会以 kwargs 形式传给 **每一个** 奖励函数 |

> 重要机制：**除了 `prompt`，数据集的所有列都会作为关键词参数传给奖励函数。**
> 所以你把答案列叫 `ground_truth`、`answer`、`label` 都行——只要奖励函数的形参同名即可。

### 5.2 奖励函数：唯一必须自己写的代码

契约（三条，缺一不可）：

```python
def my_reward(completions, ground_truth, **kwargs) -> list[float]:
    #  ① 形参：想用的字段名 = 数据集列名
    #  ② 返回：list[float]，长度 == 这一批 completion 的数量
    #  ③ 可以 async def（多个异步奖励会被并发 gather）
    return [1.0 for _ in completions]
```

可用的关键字参数（按需挑）：

| 参数 | 内容 |
|---|---|
| `prompts` | 这一批 prompt（标准格式是字符串，对话格式是 message dict） |
| `completions` | 这一批模型输出（同上，对话格式是 `[{"role","content"}]` 列表的列表） |
| `completion_ids` | token id 列表 |
| `trainer_state` | 当前训练状态（可用于课程学习） |
| `log_extra(col, values)` | 把额外列写进日志 |
| `log_metric(name, value)` | 记录自定义指标 |
| `environments` | 有 `environment_factory` 时，每个 completion 一个环境实例 |
| 数据集其他列 | 自动作为 kwargs |

### 5.3 三个真实可用的奖励函数

见 [`code/rewards.py`](code/rewards.py)：

```python
def format_reward(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    return [1.0 if re.match(pattern, c[0]["content"].strip(), re.DOTALL) else 0.0
            for c in completions]

def accuracy_reward(completions, ground_truth, **kwargs):
    ...  # 抽 <answer>，和 ground_truth 比数值

def length_penalty(completion_ids, **kwargs):
    return [-0.001 * len(ids) for ids in completion_ids]
```

> 用 token 级奖励也可以，一个最小例子（文档原文）：
> `def reward_func(completion_ids, **kwargs): return [float(len(ids)) for ids in completion_ids]` → `[2.0, 4.0]`

---

## 6. 输入输出：一次训练步的完整张量流

```
┌─ 采样阶段（推理引擎，通常是 vLLM）─────────────────────────────┐
│  input_ids:        [B, L_prompt]                              │
│  generate(num_return_sequences=G, temperature=1.0)            │
│  → completion_ids: [B*G, L_completion]                        │
│  → completion_logprobs: [B*G, L_completion]   ← 旧策略的概率   │
└───────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─ 打分阶段（你的 Python 函数，不占 GPU）─────────────────────────┐
│  rewards: [B*G]        ← 多个奖励求和 / 加权                    │
│  A_i = (r_i - mean_group) / (std_group + ε)   → advantages: [B*G] │
└───────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─ 训练阶段（前向 + 反向）────────────────────────────────────────┐
│  input_ids = cat(prompt, completion)   [B*G, L]                │
│  labels    = -100 ... -100 ⧺ completion（或带 mask 的策略梯度）│
│  forward → logits → log π_θ(y_t)                               │
│  ratio = exp(logπ_θ - logπ_old)                                │
│  loss  = -min(ratio·A, clip(ratio,1-ε,1+ε)·A).mean()           │
└───────────────────────────────────────────────────────────────┘
```

### 一个数字例子（G=4）

| 回答 | reward | 是否 > 均值(0.5) | advantage | 训练效果 |
|---|---|---|---|---|
| "…1081" | 1.0 | 是 | +1.0 | 概率 ↑ |
| "…1081" | 1.0 | 是 | +1.0 | 概率 ↑ |
| "…1000" | 0.0 | 否 | -1.0 | 概率 ↓ |
| "…1000" | 0.0 | 否 | -1.0 | 概率 ↓ |

> 如果 4 条**全对**（reward 都是 1.0），σ=0，所有 advantage=0 → **这一步什么也没学到**。
> 这就是为什么题目难度要"刚好一半会一半不会"。

---

## 7. 代码与参数

完整脚本：[`code/train_grpo.py`](code/train_grpo.py)

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from rewards import accuracy_reward, format_reward

dataset = load_dataset("json", data_files="../data/grpo_prompts.jsonl", split="train")

args = GRPOConfig(
    output_dir="./ckpt/grpo-math",
    num_generations=8,          # ★ 每个 prompt 采几条
    temperature=1.0,            # ★ 必须有随机性，否则组内全一样
    max_completion_length=512,
    loss_type="dapo",
    scale_rewards="group",
    beta=0.0,                   # 参考模型 KL 惩罚；0=不加载 ref model
    learning_rate=1e-6,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    bf16=True,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=args,
    train_dataset=dataset,
    reward_funcs=[accuracy_reward, format_reward],
)
trainer.train()
```

### 关键参数

| 参数 | 作用 | 默认 | 备注 |
|---|---|---|---|
| `num_generations` | 每个 prompt 采样条数 G | `8` | **必须整除有效 batch** |
| `temperature` | 采样温度 | `1.0` | 太低→组内无差异；太高→全是噪声 |
| `max_completion_length` | 单条回答上限 | `512` | 多轮场景=整个 episode 预算 |
| `beta` | 对 ref model 的 KL 惩罚 | `0.0` | 设 >0 会多加载一个模型 |
| `loss_type` | 损失变体 | `"dapo"` | 另有 `"grpo"`、`"dr_grpo"` 等 |
| `scale_rewards` | 优势归一化方式 | `"group"` | 用组内 std；`"batch"` 用全 batch |
| `num_iterations` | 每批数据复用几次 | `1` | >1 类似 PPO 的多 epoch |
| `reward_weights` | 多个奖励的权重 | 均等 | **只作用于 `reward_funcs`**，不影响环境奖励 |
| `mask_truncated_completions` | 截断样本是否屏蔽 | `False` | 长回答任务建议开 |
| `steps_per_generation` | 一次生成支撑几步训练 | — | 提高生成利用率 |
| `max_tool_calling_iterations` | 最多工具轮数 | — | 多轮 agent 必设 |

> 显存提示：`num_generations=8` 意味着**同一时刻要放进 8 条 completion 做前向**，
> 显存约等于 batch=8 的 SFT。0.5B 模型单卡 24GB 够用；4B 以上要考虑 `use_vllm` 或分卡。

---

## 8. 产物与验收

### 8.1 日志指标

| 指标 | 含义 | 健康表现 |
|---|---|---|
| `reward` | 平均总奖励 | 稳步上升；后期趋平 |
| `reward_std` | 组内奖励标准差 | **别长期为 0**（全对/全错就学不动） |
| `kl` | 与 ref model 的 KL | `beta=0` 时为 0；`beta>0` 时应平缓上升 |
| `entropy` | 输出熵 | 缓慢下降；骤降=塌缩 |
| `completions/mean_length` | 平均长度 | 应稳定；持续变长=在钻空子 |
| `completions/clipped_ratio` | 被截断比例 | 接近 0 |
| `tools/call_frequency` | 平均工具调用次数 | 多轮任务核心观察项 |
| `tools/failure_frequency` | 工具失败率 | 应下降 |

### 8.2 判断"RL 真的起作用"的三个检查

```
□ 留出集上准确率提升，而不只是训练 reward 提升（后者可能只是 reward hacking）
□ reward_std 仍在合理范围（0.1~0.5），说明题目难度合适
□ 输出变短或结构变清晰，而不是靠"多写点"骗分
```

### 8.3 常见 reward hacking 形态

| 现象 | 模型在干什么 | 修法 |
|---|---|---|
| 长度暴涨 | 用废话堆概率 | 加长度惩罚 / 开 `mask_truncated_completions` |
| 格式分满分但答案错 | 只学会了 `<answer>` 标签 | 提高 accuracy 权重，或去掉纯格式奖励 |
| 全部输出同一个模板 | 找到"安全但正确"的偷懒解 | 降 temperature？不，是提高题目多样性 |
| 训练 reward 上升、留出集下降 | 过拟合奖励函数 | 换更强验证器 / 加正则 |

---

## 9. 常见坑

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | `num_generations` 不整除有效 batch | 启动时报错 | 让 `batch × grad_accum × 卡数` 能被 G 整除 |
| 2 | 题目全对或全错 | `reward_std=0`，`loss≈0` | 换难度适中的题，或做难度筛选 |
| 3 | `temperature=0` | 8 条一模一样 | 保持 1.0 左右 |
| 4 | 奖励返回 numpy / tensor | 报类型错或静默出错 | 显式 `float(...)` |
| 5 | 奖励函数里改了模型状态 | 千奇百怪的 bug | 奖励函数要**纯函数**，只读输入 |
| 6 | `ground_truth` 长度对不上 | 索引错位 | 相信 TRL 的广播，别自己 `repeat` |
| 7 | 以为 `beta` 是"必须 >0" | 多花一张卡的显存 | GRPO 默认 0，先别加 |
| 8 | 用训练集奖励判断成败 | 自我感觉良好 | 一定留验证集 |

---

## 10. 练习与自测

1. **难度实验**：把 3 道题换成 3 道"远超模型能力"的题，跑 100 步，观察 `reward` 是否几乎不动。理解"全错 = 学不到"。
2. **写一个只会被 hack 的奖励**：只给 `format_reward`、不给 `accuracy_reward`，看模型多久学会"格式完美但答案乱写"。
3. **对比蒸馏**：同一批 prompt，一次用教师做蒸馏、一次用 accuracy 奖励做 GRPO，比较 200 步内验证集准确率曲线。

---

## 11. 一句话带走

> GRPO 用"组内相对分数"替掉了教师和 critic，代价是信号极稀——**一条回答只有一个数**。
> 而真实 agent 的一轮任务有几十次工具调用，只在最后给一个数太亏了。
> 下一章：让模型**在环境里行动**，每轮都拿到反馈。

---

[← 上一节：蒸馏](../02-蒸馏/README.md) · [总目录](../README.md) · [下一节：RL 环境 →](../04-RL环境/README.md)
