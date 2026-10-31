# 04 · RL 环境（RL Environments）

[← 上一节：GRPO](../03-强化学习-GRPO/README.md) · [总目录](../README.md) · [下一节：Harness 与规模化 →](../05-Harness与规模化/README.md)

---

## 1. 一句话定义

**把"一个分数"从"评一句话"升级成"评一整个在环境里行动的过程"：模型可以调工具、看到结果、再决定下一步，直到环境判定它做成了没有。**

第 3 步的奖励函数只看模型说了什么；环境看模型**做成了什么**。

---

## 2. 原文译文 · 4. RL environments

> 奖励来自**在一个环境中行动**——这个环境保持状态、暴露工具、并在多轮交互中回应你。环境负责执行工具、跟踪发生了什么变化、并检查最终结果，例如运行一个编程任务的测试。
>
> 每个 episode 都比单个回答更长，所以**生成 rollout 成了训练中最慢的部分**。

### 原文推荐资源

| 类型 | 链接 |
|---|---|
| Watch | [Training Agents 4: from reward functions to environments](https://www.youtube.com/watch?v=nJV3yUuz6DU)（[合集](https://huggingface.co/collections/sergiopaniego/training-agents-class-4-rl-environments-6aa27543918888a0f2b5312c)） · [RL for Agents Workshop](https://www.youtube.com/watch?v=cixmqTsi2A4) |
| Read | [Why RL environments work now (and could not in 2016)](https://huggingface.co/blog/sergiopaniego/rl-environments-story) |
| Projects | [Paint watercolours with code](https://huggingface.co/blog/train-to-paint-with-code) · [Autonomous driving with CARLA](https://huggingface.co/blog/sergiopaniego/bringing-carla-to-openenv-trl) |
| Docs | [TRL + OpenEnv](https://huggingface.co/docs/trl/openenv) · [OpenEnv docs](https://huggingface.co/docs/openenv) |

---

## 3. 奖励函数 vs 环境：差在哪

| | 第 3 步：奖励函数 | 第 4 步：环境 |
|---|---|---|
| 模型能行动几次 | 1 次（写一条回答） | 多次（多轮工具调用） |
| 谁执行工具 | 没有人，工具只是"写在文本里" | **环境真的执行** |
| 状态 | 无 | 有（文件系统、数据库、盘面……） |
| 反馈粒度 | 一次打分 | 每轮都有观测（工具结果/报错） |
| 分数从哪来 | `reward_funcs` 返回 | `env.get_reward()` |
| 长度单位 | 一条 completion | 一个 episode（几十轮） |
| 主要瓶颈 | 采样 | **rollout 太慢**（原文强调） |

> 一句话：奖励函数是**阅卷老师**，环境是**考场**。

---

## 4. 一个 episode 到底怎么跑

```
   prompt（数据集里的，或 env.reset() 返回的）
      │
      ▼
   轮 1  模型：先读文件 → tool_call read_file("solution.py")
         环境：返回文件内容
   轮 2  模型：除零没处理 → tool_call write_file(...)
         环境：写入磁盘，返回"已写入"
   轮 3  模型：tool_call run_tests()
         环境：跑 pytest，返回 "2 passed"
   轮 4  模型：修好了 → 结束
      │
      ▼
  env.get_reward() → 1.0
```

**几个容易被忽略的点**：

1. 每一轮的 **prompt + completion + 工具结果都是 token**，全都算进 `max_completion_length` 这个总预算。
2. 工具抛异常不是崩溃，而是**变成一条工具结果**回喂模型（"文件不存在"、"语法错误"）。模型因此学会读报错。
3. 一个 episode 结束的方式有三种：模型自己停、撞到 token 预算、撞到 `max_tool_calling_iterations`。
4. **组内 8 条 rollout 的初始状态可能不同**（如果 `reset()` 自己采样子任务），此时"组内比较"要更小心。

---

## 5. 接口契约：写一个环境类

TRL 的约定非常具体，逐条照做就不会踩坑：

| 要求 | 说明 | 不满足会怎样 |
|---|---|---|
| 类定义在**模块顶层** | 第 5 步要 pickle 到 rollout 进程 | `TypeError: can't pickle` |
| `__init__(self)` **不带参数** | 参数通过 `reset()` 的 kwargs 传 | factory 调用失败 |
| `reset(**kwargs) -> str \| None` | 必需。返回观测字符串（当 prompt）或 `None`（用数据集的 prompt） | 训练无法开始 |
| 公开方法 = 工具 | 方法名即工具名，**docstring 即 description**，类型注解即 schema | 模型不知道有这个工具 |
| `get_reward() -> float` | 可选。定义了就会与 `reward_funcs` 的分数**相加** | 只能用外部奖励 |
| 别写通用的 `step(action)` | 工具应该是**具体的**（`read_file` 而不是 `step`） | 模型学不到动作语义 |
| 工具内部状态要维护 | 环境自己记住改了什么 | 多轮之间状态丢失 |

```python
class MyEnv:
    def __init__(self):
        self.state = None

    def reset(self, **kwargs):
        self.state = ...
        return "任务描述……"      # 或 None

    def my_tool(self, arg: str) -> str:
        """一句话说明这个工具干什么。

        Args:
            arg: 参数说明（schema 会从这里生成）
        """
        return "结果"

    def get_reward(self) -> float:
        return 1.0 if done else 0.0
```

---

## 6. 输入格式

环境有两种数据驱动方式。

### 方式 A：环境自采样（无数据集）

`reset()` 内部随机生成任务，返回 prompt。**连 train_dataset 都不用给。**

```python
training_args = GRPOConfig(environment_factory=IncrementEnv, ...)
trainer = GRPOTrainer(model=..., args=training_args)      # 没有 train_dataset
```

### 方式 B：数据集驱动（推荐，任务可控）

`prompt` 列照旧，**其余列会作为 kwargs 传给 `reset()`**。

```json
{
  "prompt": "solution.py 里的测试失败了，请修好它。",
  "buggy_code": "def divide(a, b):\n    return a / b\n",
  "tests": "from solution import divide\n\ndef test_zero():\n    ..."
}
```

| 列 | 去向 |
|---|---|
| `prompt` | 第一条用户消息 |
| `buggy_code` / `tests` | `env.reset(buggy_code=..., tests=...)` |
| `environment` | 有多个 factory 时用来路由（见下） |

### 多个环境并存

```python
training_args = GRPOConfig(
    environment_factory={"increment": IncrementEnv, "coding": CodingEnv},
    ...
)
```

数据集里 `environment` 列填 `"increment"` 或 `"coding"`，TRL 按名字选 factory。

---

## 7. 输入输出：多轮视角

一个 episode 就是一条很长的 input_ids（受 `max_completion_length` 约束）：

| 段 | prompt | turn 1 生成 | 工具结果 | turn 2 生成 | … |
|---|---|---|---|---|---|
| `loss_mask` | `-100` | **保留** | `-100` | **保留** | `-100` |

也就是说：**模型自己说的算 loss，模型看到的不算**——和 SFT 的掩码规则同源，只是这里要跨好几轮。

```
整条 episode 的 logprobs  ──►  env.get_reward()  →  reward [1]
                                      │
                                      ▼
                        与另外 7 条 rollout 组内标准化 → advantage
                                      │
                                      ▼
                        对每一轮「保留」的 token 做策略梯度
```

**和 SFT 的 mask 规则完全一致**：只有 assistant（模型自己说的）算 loss，工具结果和环境返回一律 `-100`。
区别只是这条序列由环境动态拼出来，不是数据里写好的。

---

## 8. OpenEnv：环境的标准容器

原文把 OpenEnv 作为环境生态的标准。它的关键点：

| 特性 | 含义 |
|---|---|
| **Gymnasium 风格 API** | `reset()` / `step()` / `state()` —— 和你熟悉的 RL 接口一致 |
| **WebSocket 传输** | 客户端与服务器分离，环境可以跑在另一台机器/云端 |
| **同步 + 异步** | 天然适配异步 rollout |
| **每个环境 = 一个 Docker 镜像** | 依赖、工具、状态全打包，本地/HF Space/云沙箱都能跑 |
| **环境目录（catalog）** | 现成的环境可以直接拿来训，不用自己写 |
| **集成** | TRL、Unsloth、SkyRL、ART、Oumi、torchforge、Miles |
| **Harbor** | 在环境里跑 Claude Code / Codex / OpenCode 这类真实 agent，并抓下 token（第 5 步的基础） |

### 用现成环境的最小例子（文档原文的 Echo 环境）

```python
from envs.echo_env import CallToolAction, EchoEnv

class EchoToolEnv:
    def __init__(self):
        self.env = EchoEnv(base_url="https://openenv-echo-env.hf.space")
        self.reward = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs.observation.echoed_message

    def call_tool(self, message: str) -> str:
        """调用 Echo 环境里的工具。"""
        obs = self.env.step(CallToolAction(message=message))
        self.reward = obs.observation.reward
        return obs.observation.echoed_message

    def get_reward(self):
        return self.reward

def reward_func(environments, **kwargs):
    return [env.reward for env in environments]

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=GRPOConfig(output_dir="echo", chat_template_kwargs={"enable_thinking": False},
                    log_completions=True, ...),
    environment_factory=EchoToolEnv,
    reward_funcs=[reward_func],
)
```

> 有 `environment_factory` 时，奖励函数的签名里会出现 `environments` 参数——
> 每个 completion 对应一个环境实例，所以能直接读 `env.reward`。

---

## 9. 代码与参数

- 环境类：[`code/env_factory.py`](code/env_factory.py)（`IncrementEnv` 最小示例 + `CodingEnv` 编程环境）
- 训练脚本：[`code/train_grpo_env.py`](code/train_grpo_env.py)

```python
args = GRPOConfig(
    output_dir="./ckpt/grpo-coding-env",
    environment_factory=CodingEnv,   # ★ 唯一的关键改动
    num_generations=8,
    max_completion_length=4096,      # ★ 整个 episode 的 token 预算
    max_tool_calling_iterations=20,
    log_completions=True,            # ★ 调试环境时必开
    learning_rate=1e-6,
)
trainer = GRPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", args=args,
                      train_dataset=dataset)   # reward_funcs 可为空
```

### 参数对照（与第 3 步的差异）

| 参数 | 第 3 步（奖励函数） | 第 4 步（环境） |
|---|---|---|
| `max_completion_length` | 一条回答的长度 | **整个 episode** 的预算，要放大 5–20 倍 |
| `max_tool_calling_iterations` | 不用设 | 必设（防止无限调工具） |
| `log_completions` | 可选 | **强烈建议开**，否则环境行为是黑盒 |
| `reward_funcs` | 必需 | 可为空（分数来自 `get_reward`） |
| `train_dataset` | 必需 | 可省（环境自采样） |

### 环境服务器

如果环境跑在 OpenEnv 服务里（Docker 镜像 + WebSocket），注意并发上限：

```bash
MAX_CONCURRENT_ENVS=9 openenv serve --port 8000
```

并发数至少要覆盖 `num_generations × 同时在跑的 batch`，否则 rollout 会排队，训练变慢。

---

## 10. 产物与验收

### 10.1 日志指标

| 指标 | 含义 | 健康表现 |
|---|---|---|
| `reward` | 二元通过率（本例） | 从接近 0 稳步上升 |
| `reward_std` | 组内差异 | 太低说明题目太简单/太难 |
| `tools/call_frequency` | 每 episode 平均工具调用数 | 从 1–2 涨到 5–10 说明学会用工具 |
| `tools/failure_frequency` | 工具失败率 | 应下降（学会正确传参） |
| `completions/mean_length` | episode 平均 token 数 | 稳定；暴涨=在兜圈子 |
| `completions/clipped_ratio` | 撞预算上限比例 | 应 < 5% |
| `entropy` | 输出熵 | 平缓下降 |

### 10.2 看 `log_completions` 的完整 episode

这是**最有效的验收方式**。一个训练良好的 episode 应该像：

```
[turn 1] read_file("solution.py")            → 文件内容
[turn 2] "除零没抛异常"                        ← 有推理
[turn 3] write_file("solution.py", "...")    → 已写入
[turn 4] run_tests()                          → 2 passed
[turn 5] "在 divide 里加除零检查，测试通过"      ← 简洁总结
```

对比训练前的坏样子：

```
[turn 1] write_file("solution.py", "def divide(...): ...")   ← 不看就改
[turn 2] run_tests() → 1 failed 3 passed
[turn 3] write_file(...)  ← 反复瞎试，撞预算
```

### 10.3 产物

- `./ckpt/grpo-coding-env/`：会使用工具、会读报错的模型
- 一个可复用的环境（下一步 Harness 会把它包进真实 agent）

---

## 11. 常见坑

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | 工具没有 docstring | 模型不知道工具有什么用 | 写 docstring + 类型注解 |
| 2 | `max_completion_length` 还是 512 | 第 2 轮就被截断，永远失败 | 按 episode 长度放大 |
| 3 | 奖励塑形过度 | 模型刷中间步骤 | 优先用二元奖励（文档明确建议） |
| 4 | 环境在本机跑 `exec` | 安全风险 | 用 Docker / 远程沙箱 |
| 5 | 环境很慢 | GPU 空转等 rollout | 提高 `MAX_CONCURRENT_ENVS`、开 `use_vllm` |
| 6 | 不同 rollout 初始状态差异大 | 组内比较失去意义 | 让 `reset()` 采样同一难度，或按难度分组 |
| 7 | `reset()` 返回空字符串 | prompt 为空，模型乱说 | 返回明确任务描述或 `None` |
| 8 | 状态没在 `reset` 里清干净 | 上一个 episode 的污染 | 所有状态在 `reset` 里重置 |
| 9 | 忘了 `get_reward` 的调用时机 | 分数在动作中途就变了 | 只在终结时计算 |

---

## 12. 练习与自测

1. **最小环境**：把 `IncrementEnv` 的目标改成"从 10 减到 0"（加一个 `decrement` 工具），训练 100 步，看 `tools/call_frequency` 是否能稳定到 10。
2. **错误反馈**：在 `read_file` 里故意对不存在的文件抛 `FileNotFoundError`，观察模型是否学会"先列文件再读"。
3. **二元 vs 塑形**：给 `CodingEnv.get_reward()` 加一个"改了文件就给 0.2"的塑形分，对比 200 步后真实通过率——你大概率会看到塑形版本更差。

---

## 13. 一句话带走

> 环境把 RL 从"评一句话"变成了"评一次行动"，好评如潮，但**每个 episode 都要真的跑起来**——
> 于是 rollout 成了瓶颈，而模型还是在你写的循环里跑。
> 下一章：干脆**别自己写循环**，让真实 agent harness 来跑。

---

[← 上一节：GRPO](../03-强化学习-GRPO/README.md) · [总目录](../README.md) · [下一节：Harness 与规模化 →](../05-Harness与规模化/README.md)
