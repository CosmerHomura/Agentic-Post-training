# 99 · 附录：全文对照、资源总表、术语、速查

[← 上一节：Harness 与规模化](../05-Harness与规模化/README.md) · [总目录](../README.md)

---

## 附录 A · 原文与译文全文对照

> 原文：<https://huggingface.co/blog/huggingface/train-agents>
> 说明：为方便对照，段落逐段对齐；原文中每节的 Watch / Read / Docs 链接统一收在「附录 B」。

### A.0 标题与引言

**原文**
> A curriculum for post-training agents
>
> Post-training refers to the training stages that follow pretraining, such as supervised fine-tuning, distillation and reinforcement learning. These stages turn a pretrained model into one that follows instructions, uses tools or solves a specific task.
>
> When a model acts as an agent, it works over many turns. It calls tools, reads the results and decides what to do next. Agentic post-training applies these stages to agents, so training has to account for the whole sequence of turns instead of a single answer.

**译文**
> 一份面向 agent 的后训练课程
>
> 后训练指的是预训练之后的那些训练阶段，例如监督微调、蒸馏和强化学习。这些阶段把一个预训练模型变成会遵循指令、会使用工具、或能解决特定任务的模型。
>
> 当模型作为 agent 工作时，它要在很多轮里行动：调用工具、读取结果、决定下一步做什么。Agentic 后训练就是把这些阶段用在 agent 上——于是训练必须考虑**整条多轮交互的序列**，而不是单个回答。

**原文**
> This page is organized in five steps, after an introductory section. Every step includes a diagram of the method, a livestream or slides, a blog post that explains the idea and the relevant documentation. Some steps add example projects built with that method. The table below summarizes, for each step, what the model learns from, the TRL trainer and what you need to prepare.
>
> The order of the steps follows Training Agents, a livestream series on the Hugging Face YouTube channel. Each session covers one post-training method, with training a coding agent as the running example. The Post-training Agents playlist has the four sessions plus two related workshops.

**译文**
> 本页在导言之后组织为五个步骤。每一步都包含：方法示意图、一场直播或幻灯片、一篇解释思路的博客，以及相关文档。部分步骤还附了用该方法构建的示例项目。下面的表格总结了每一步：模型从什么信号中学习、用哪个 TRL trainer、以及你需要准备什么。
>
> 这些步骤的顺序沿袭 Hugging Face YouTube 频道的直播系列 **Training Agents**。每一期讲一种后训练方法，并始终以**训练一个编程 agent** 作为贯穿全程的示例。**Post-training Agents** 播放列表包含这四期节目以及两场相关 workshop。

### A.1 Steps at a glance

**原文**
> | Step | The model learns from | TRL trainer | What you need |
> | 1. SFT | agent traces from a stronger model | SFTTrainer | a dataset of agent traces |
> | 2. Distillation | a teacher's tokens or probabilities | DistillationTrainer | prompts and a teacher model |
> | 3. RL | a score for each answer | GRPOTrainer | prompts and a reward function |
> | 4. RL environments | the outcome of acting in an environment | GRPOTrainer | tasks and an environment (OpenEnv) |
> | 5. Harnesses and scale | full runs of a real agent | AsyncGRPOTrainer | tasks, an environment and an agent harness |

**译文**

| 步骤 | 模型从什么学习 | TRL trainer | 你需要准备 |
|---|---|---|---|
| 1. SFT | 来自更强模型的 agent 轨迹 | `SFTTrainer` | 一个 agent 轨迹数据集 |
| 2. 蒸馏 | 教师的 token 或概率 | `DistillationTrainer` | prompts + 一个教师模型 |
| 3. RL | 每个回答的得分 | `GRPOTrainer` | prompts + 一个奖励函数 |
| 4. RL 环境 | 在环境中行动的结果 | `GRPOTrainer` | 任务 + 一个环境（OpenEnv） |
| 5. Harness 与规模化 | 真实 agent 的完整运行 | `AsyncGRPOTrainer` | 任务 + 环境 + 一个 agent harness |

### A.2 · 0. Before you start

**原文**
> The terms used in the rest of the page (agent, harness, scaffold) and an overview of all the steps.

**译文**
> 本页后续会用到的术语（agent、harness、scaffold），以及所有步骤的总览。

### A.3 · 1. Supervised fine-tuning (SFT)

**原文**
> The model learns to copy agent traces, recordings of an agent solving a task with every tool call and its result, usually produced by a stronger model. It's cheap and stable.
>
> The model only sees correct traces, never its own attempts. When it makes a mistake while solving a task, it has not learned how to recover from it. On-policy distillation and RL address this by training on the model's own outputs.

**译文**
> 模型学会复制 **agent 轨迹**——也就是 agent 解决一个任务的过程记录，其中包含每一次工具调用及其结果，通常由更强的模型产生。这一步便宜且稳定。
>
> 模型只会见到**正确的**轨迹，永远见不到它自己的尝试。当它在解题过程中犯了错，它并没有学过如何从错误中恢复。**On-policy 蒸馏**和 **RL** 通过在模型自己的输出上训练来解决这个问题。

### A.4 · 2. Distillation

**原文**
> A teacher model supervises the student, either through the data it writes or through its probabilities on the student's own tokens. The teacher is usually a larger model, several expert models distilled into one student (as in MOPD), or the same model with extra context in the case of self-distillation. The signal is dense, one value per token, so training needs fewer samples than RL.
>
> On the other hand, the student is limited by what the teacher knows, and on-policy distillation has to run the teacher at every training step. RL replaces the teacher with a reward that only checks the result.

**译文**
> 由一个教师模型来监督学生：监督可以来自**教师写出来的数据**，也可以来自**教师在学生自己那些 token 上的概率**。教师通常是一个更大的模型，也可以是"多个专家模型蒸馏进一个学生"（例如 MOPD），或者在自蒸馏的情况下是"同一个模型 + 更长的上下文"。这个信号是**密集的：每个 token 一个值**，所以训练所需的样本量比 RL 少。
>
> 反过来，学生会被教师的知识水平所限制；而在 on-policy 蒸馏里，**每一个训练步都必须跑一次教师**。RL 则用"只看结果的奖励"替代了教师。

### A.5 · 3. Reinforcement learning (RL)

**原文**
> In reinforcement learning, the model learns from a score on its own outputs. It writes an answer, a reward function scores it, and training makes high-scoring answers more likely. This removes the need for a teacher. A way to check the answer is enough.
>
> The algorithm used in this step is GRPO (Group Relative Policy Optimization). For each prompt it samples several answers, scores them all, and raises the probability of the ones that scored above the group average. Comparing answers within the group tells the model which attempts were better, without training a separate model to estimate that.
>
> The signal is weaker than in distillation, one number for the whole answer instead of one per token, so the model needs many attempts to learn. With a plain reward function each answer is a single turn, scored once. Environments let the model act several times and see the result of each action.

**译文**
> 在强化学习中，模型从**对自己输出的一个分数**里学习。它写一个答案，奖励函数给它打分，训练让高分答案变得更可能。这样就**不再需要教师**——只要能校验答案就够了。
>
> 这一步用的算法是 **GRPO（Group Relative Policy Optimization）**。对每个 prompt，它采样若干个答案，给它们全部打分，然后提高那些**得分高于组内平均**的答案的概率。在组内比较答案，就告诉了模型哪些尝试更好，**而不需要另外训练一个模型来估计这件事**。
>
> 这个信号比蒸馏弱：**整条答案只有一个数**，而不是每个 token 一个值，所以模型需要很多次尝试才能学到东西。用普通奖励函数时，每个答案只有一轮、只打一次分。而**环境**让模型可以行动多次，并看到每个动作的结果。

### A.6 · 4. RL environments

**原文**
> The reward comes from acting in an environment that keeps state, exposes tools and answers back over several turns. The environment runs the tools, keeps track of what changed and checks the final result, for example by running the tests of a coding task.
>
> Each episode is longer than a single answer, so generating rollouts becomes the slowest part of training.

**译文**
> 奖励来自**在一个环境中行动**——这个环境保持状态、暴露工具、并在多轮交互中回应你。环境负责执行工具、跟踪发生了什么变化、并检查最终结果，例如运行一个编程任务的测试。
>
> 每个 episode 都比单个回答更长，所以**生成 rollout 成了训练中最慢的部分**。

### A.7 · 5. Harnesses and scale

**原文**
> Each rollout is a full agent run through an existing agent harness inside a remote sandbox. Training through the same harness the agent uses in practice means the model learns in the setting where it will run.
>
> Generation and training run on separate GPUs, so long episodes don't leave the trainer idle. The trainer then sends the updated weights back to vLLM.

**译文**
> 每一次 rollout 都是**一个完整的 agent 运行**——通过一个现成的 agent harness、在一个远程沙箱里跑完。用 agent 实际部署时会用的同一个 harness 来训练，意味着模型是在**它将来真正运行的那个环境**里学习。
>
> 生成与训练跑在**不同的 GPU** 上，所以很长的 episode 不会让训练器干等着。随后训练器把更新后的权重送回 vLLM。

---

## 附录 B · 资源总表（原文全部链接）

### B.0 通用

| 资源 | 链接 |
|---|---|
| 课程页 | <https://huggingface.co/blog/huggingface/train-agents> |
| 播放列表 Post-training Agents | <https://www.youtube.com/playlist?list=PLo2EIpI_JMQvQZm-kVlz4wY1vWF0LBcf5> |
| 术语博客 | <https://huggingface.co/blog/agent-glossary> |
| 总览幻灯片（20 页） | <https://huggingface.co/spaces/sergiopaniego/train-the-agent-slides> |
| LLM Course | <https://huggingface.co/learn/llm-course> |

### B.1 SFT

| 类型 | 链接 |
|---|---|
| 图 | <https://huggingface.co/buckets/merve/post-training-curriculum/resolve/1_sft.png?download=true> |
| 直播 | <https://www.youtube.com/watch?v=rNgUoH7Wbv8> |
| 直播代码 | <https://huggingface.co/burtenshaw/gemma4-pi-mono-youtube-livestream-1-scripts> |
| 阅读 | <https://x.com/ben_burtenshaw/status/2067615361428545566> |
| 文档 | <https://huggingface.co/docs/trl/sft_trainer> |

### B.2 蒸馏

| 类型 | 链接 |
|---|---|
| 图 | <https://huggingface.co/buckets/merve/post-training-curriculum/resolve/2_distillation.png?download=true> |
| 直播 | <https://www.youtube.com/watch?v=GZT_oNQnLV4> |
| 合集 | <https://huggingface.co/collections/sergiopaniego/training-agents-class-2-distillation-6a47a62b9f09867dccf61e1e> |
| 阅读 1 | <https://huggingface.co/blog/sergiopaniego/brief-history-of-distillation-in-ai> |
| 阅读 2 | <https://huggingface.co/blog/sergiopaniego/distillation-2026> |
| 文档 | <https://huggingface.co/docs/trl/distillation_trainer> |

### B.3 GRPO

| 类型 | 链接 |
|---|---|
| 图 | <https://huggingface.co/buckets/merve/post-training-curriculum/resolve/3_rl.png?download=true> |
| 直播 | <https://www.youtube.com/watch?v=ztdTed5egrM> |
| 合集 | <https://huggingface.co/collections/sergiopaniego/training-agents-class-3-reinforcement-learning-6ad6aa71050342629dd0e3ec> |
| 阅读 | <https://huggingface.co/blog/sergiopaniego/agentic-rl-2026> |
| 项目 | <https://huggingface.co/blog/grpo-with-trl-ifstruct> |
| 文档 | <https://huggingface.co/docs/trl/grpo_trainer> |

### B.4 RL 环境

| 类型 | 链接 |
|---|---|
| 图 | <https://huggingface.co/buckets/merve/post-training-curriculum/resolve/4_environments.png?download=true> |
| 直播 | <https://www.youtube.com/watch?v=nJV3yUuz6DU> |
| 合集 | <https://huggingface.co/collections/sergiopaniego/training-agents-class-4-rl-environments-6aa27543918888a0f2b5312c> |
| Workshop | <https://www.youtube.com/watch?v=cixmqTsi2A4> |
| 阅读 | <https://huggingface.co/blog/sergiopaniego/rl-environments-story> |
| 项目 1 | <https://huggingface.co/blog/train-to-paint-with-code> |
| 项目 2 | <https://huggingface.co/blog/sergiopaniego/bringing-carla-to-openenv-trl> |
| 文档 · TRL+OpenEnv | <https://huggingface.co/docs/trl/openenv> |
| 文档 · OpenEnv | <https://huggingface.co/docs/openenv> |

### B.5 Harness 与规模化

| 类型 | 链接 |
|---|---|
| 图 | <https://huggingface.co/buckets/merve/post-training-curriculum/resolve/5_scale.png?download=true> |
| 幻灯片 | <https://huggingface.co/spaces/FineEnvs/multi-harness-rl-slides> |
| 阅读 1 | <https://huggingface.co/spaces/FineEnvs/multi-harness-rl> |
| 阅读 2 | <https://huggingface.co/blog/sergiopaniego/rl-environments-2026> |
| 阅读 3 | <https://huggingface.co/blog/async-rl-training-landscape> |
| 文档 · AsyncGRPO | <https://huggingface.co/docs/trl/async_grpo_trainer> |
| 文档 · OpenEnv harnesses | <https://huggingface.co/docs/openenv/tutorials/harnesses> |

---

## 附录 C · 术语表

| 英文 | 中文 | 定义 | 在哪一章出现 |
|---|---|---|---|
| post-training | 后训练 | 预训练之后的所有训练阶段（SFT / 蒸馏 / RL） | 全篇 |
| agentic post-training | Agent 后训练 | 把后训练方法用在多轮、会调工具的模型上 | 全篇 |
| trace | 轨迹 | 一次任务求解的完整记录（含工具调用与结果） | 第 1 章 |
| agent trace | agent 轨迹 | 上面那种记录，作为 SFT 的监督数据 | 第 1 章 |
| rollout | 采样 / 一条 rollout | 一次完整的生成过程；多轮时 = 一个 episode | 第 4、5 章 |
| episode | 回合 | 从 `reset()` 到结束的一整段交互 | 第 4 章 |
| harness | 外部壳 / 驱动框架 | 包在模型外、决定"下一步做什么"的程序 | 第 0、5 章 |
| scaffold | 脚手架 | 与 harness 近义，偏"模型之外的工程结构" | 第 0 章 |
| environment | 环境 | 有状态、能执行动作、能判定的世界 | 第 4 章 |
| on-policy | 同策略 | 用当前模型自己生成的数据训练 | 第 2、3 章 |
| off-policy | 异策略 | 用稍旧策略生成的数据训练（异步的必然结果） | 第 5 章 |
| teacher / student | 教师 / 学生 | 蒸馏里提供信号的模型 / 被训练的模型 | 第 2 章 |
| GKD | 广义知识蒸馏 | 在模型自己生成的序列上做蒸馏的方法 | 第 2 章 |
| MOPD | 多专家蒸馏 | 把多个专家模型的分布蒸馏进一个学生 | 第 2 章 |
| self-distillation | 自蒸馏 | 教师=同一模型+更多上下文 | 第 2 章 |
| GRPO | 组相对策略优化 | 组内比较优势、无需 critic 的 RL 算法 | 第 3 章 |
| advantage | 优势 | 某条回答比组内平均好多少 | 第 3 章 |
| reward function | 奖励函数 | 输入一批输出、返回一批分数的 Python 函数 | 第 3 章 |
| reward hacking | 奖励作弊 | 模型找到骗分捷径而非真解题 | 第 3 章 |
| format reward | 格式奖励 | 只检查输出结构是否合规的奖励 | 第 3 章 |
| KL penalty | KL 惩罚 | 限制新策略偏离参考模型的程度 | 第 3 章 |
| critic | 价值网络 | PPO 里估计状态价值的模型；GRPO 省掉了它 | 第 3 章 |
| staleness | 陈旧度 | 样本相对当前权重落后了几次更新 | 第 5 章 |
| weight sync | 权重同步 | trainer 把新权重推给推理服务 | 第 5 章 |
| loss mask | 损失掩码 | 标记哪些 token 参与 loss | 第 1、5 章 |
| tool call | 工具调用 | 模型输出结构化的函数调用请求 | 第 1、4 章 |
| sandbox | 沙箱 | 隔离的执行环境（容器 / 远程 VM） | 第 5 章 |
| Harbor | Harbor | OpenEnv 里负责起沙箱、跑 harness、抓 trace 的服务 | 第 5 章 |

---

## 附录 D · 五步速查表

| | ① SFT | ② 蒸馏 | ③ GRPO | ④ 环境 | ⑤ Harness |
|---|---|---|---|---|---|
| Trainer | `SFTTrainer` | `DistillationTrainer` | `GRPOTrainer` | `GRPOTrainer` | `AsyncGRPOTrainer` |
| 数据集列 | `messages`(+`tools`) | `prompt` | `prompt`+任意列 | `prompt`+环境参数 | `prompt`+任务配置 |
| 模型产出 | 每个 token 分布 | 自己生成 + 两组分布 | G 条回答 | 多轮 action | 真实 agent 全程 |
| 监督 | 硬标签 | 教师分布 | 一个分数 | 一个结果 | 一个结果 |
| 损失 | 交叉熵 | JSD/KL | clipped PG | clipped PG | clipped PG |
| 关键参数 | `assistant_only_loss` | `beta` | `num_generations` | `max_completion_length` | `vllm_server_url` |
| 核心指标 | `mean_token_accuracy` | `completions/clipped_ratio` | `reward`/`reward_std` | `tools/call_frequency` | `reward` + 墙钟时间 |
| 最小卡数 | 1 | 1（学生+教师） | 1 | 1 | 2 |

---

## 附录 E · FAQ

**Q1：我只有一张 16GB 卡，能学完吗？**
能。前三步用 `Qwen3-0.6B` / `Qwen2.5-0.5B` 全参 SFT 也能跑；蒸馏用 0.5B 学生 + 1.5B 教师（开 `gradient_checkpointing`）；GRPO 把 `num_generations` 降到 4、`max_completion_length` 降到 512。第 4 步能跑。第 5 步需要两块卡——可以用 Colab / 免费额度跳过，只读代码。

**Q2：SFT 数据从哪来？**
三个来源：① 用更强的模型（Claude / GPT / Qwen-Max）跑你的 agent，把 transcript 存下来；② 人类专家演示；③ 你自己写脚本合成（前提是有可验证的执行环境，否则质量难保证）。关键是**必须是完整 transcript**，包括工具调用和结果。

**Q3：为什么 SFT 之后还要 RL？SFT 不能一步到位吗？**
SFT 只见过正确轨迹，所以遇到自己犯错时不会恢复；它优化的是"和示范的相似度"，不是"任务成功率"。RL 直接优化成功率，能发现示范里没有的解法。原文的比喻很准确：SFT 学"抄"，RL 学"练"。

**Q4：GRPO 的 `reward` 一直不涨怎么办？**
按这个顺序排查：① `reward_std` 是否为 0（全对/全错）；② `temperature` 是否太低（组内全一样）；③ 奖励函数是否有 bug（打印几条 `completions` + `rewards` 看）；④ 题目难度是否合适；⑤ `learning_rate` 是否太大（loss 爆炸）或太小。

**Q5：怎么发现 reward hacking？**
三个信号：`completions/mean_length` 持续上涨；训练 `reward` 涨但留出集不涨；抽看 transcript 发现"格式完美、内容空洞"。对策：加长度惩罚、提高准确率权重、用更强的验证器、定期人工抽检。

**Q6：`beta` 到底设多少？**
蒸馏的 `beta` 是散度选择器（0=前向 KL，1=反向 KL，0.5=JSD），先试 0.5。GRPO 的 `beta` 是参考模型 KL 惩罚，默认 0.0（不加载 ref model，省显存），想做稳定性约束再设 0.01~0.04。**两个 beta 完全不是一回事。**

**Q7：环境里的 `max_completion_length` 为什么老是截断？**
因为它管的是**整个 episode** 的 token 预算，不是单轮。一个 10 轮的工具调用任务轻松超过 8k token。要同时调大 `max_completion_length` 和 `max_tool_calling_iterations`，并看 `completions/clipped_ratio`。

**Q8：第 5 步报 `TypeError: can't pickle` 怎么办？**
把奖励函数/工具/环境类挪到模块顶层（不要 lambda、不要闭包、不要 `__main__` 里的嵌套函数）。需要预绑定参数就 `functools.partial(fn, arg=...)`。工具函数名会作为注册键，所以别重名。

**Q9：可以跳过前面直接做第 5 步吗？**
技术上可以（`AsyncGRPOTrainer` 不依赖前面的产物），但实践上不建议：模型连工具调用格式都不会时，真实 harness 里的 rollout 几乎全是失败，`reward_std≈0`，什么也学不到。**先用 SFT 把格式教会，是性价比最高的一步。**

**Q10：这套流程和"预训练"是什么关系？**
预训练学语言与世界知识；后训练学"怎么用这些知识完成任务"。五步全在后训练范畴内，都不改变模型规模，只改行为分布。

---

[← 上一节：Harness 与规模化](../05-Harness与规模化/README.md) · [总目录](../README.md)
