# 00 · 开始之前：术语、全景图、环境

[← 总目录](../README.md) · [下一节：SFT →](../01-监督微调-SFT/README.md)

---

## 1. 这一章解决什么

原文在正式讲五步之前，先让你搞清"我们在训练什么"。
如果 *agent / harness / scaffold / environment / rollout / trace* 这几个词在你脑子里是一团，后面的五步会全部糊掉。这一章先把它们钉死，再画出整条流水线。

---

## 2. 原文译文 · 引言

> 后训练指的是预训练之后的那些训练阶段，例如监督微调、蒸馏和强化学习。这些阶段把一个预训练模型变成会遵循指令、会使用工具、能解决特定任务的模型。
>
> 当模型作为 agent 工作时，它要在很多轮里行动：调用工具、读取结果、决定下一步做什么。**Agentic 后训练**就是把这些阶段用在 agent 上——于是训练必须考虑整条多轮交互的序列，而不是单个回答。

> 本页组织为五个步骤，前面有一段导言。每一步都包含：方法示意图、一场直播或幻灯片、一篇解释思路的博客，以及相关文档。部分步骤还附了用该方法构建的示例项目。下面的表格总结了每一步：模型从什么信号中学习、用哪个 TRL trainer、你需要准备什么。
>
> 这些步骤的顺序沿袭直播系列 **Training Agents**（Hugging Face 的 YouTube 频道）。每一期讲一种后训练方法，并始终以"训练一个编程 agent"作为贯穿全程的示例。播放列表 [Post-training Agents](https://www.youtube.com/playlist?list=PLo2EIpI_JMQvQZm-kVlz4wY1vWF0LBcf5) 包含四期节目加两场相关 workshop。

> **原文「Before you start」**：如果你对 agent 不熟，建议先读 *[Harness, Scaffold, and the AI Agent Terms Worth Getting Right](https://huggingface.co/blog/agent-glossary)*。如果你对后训练整体还不熟，[LLM Course](https://huggingface.co/learn/llm-course) 覆盖了这些方法背后的基础。想看总览，就看 *Train the agent, not just the model* 幻灯片。

---

## 3. 六个词，一次说清

教程里所有混乱都来自这六个词。下面是我给的"人话定义 + 反例"。

| 词 | 人话定义 | 反面例子（别这么理解） |
|---|---|---|
| **agent** | 能**多轮**调用工具、看结果、再决定下一步的模型 | 一次性吐一段长文本的聊天模型（那是 chat，不是 agent） |
| **harness**（外部壳/脚手架） | 驱动 agent 的那套程序：决定何时调工具、怎么拼上下文、什么时候停 | 不是模型的一部分，是包在模型外面的循环 |
| **scaffold** | 与 harness 近义，更偏"模型之外的工程结构" | 不是训练框架（TRL 是 trainer，不是 scaffold） |
| **environment** | 有状态的"世界"：能执行动作、记住变化、最后给结果 | 不是一个静态数据集；它有 `reset()` 和 `step()` |
| **rollout** | 一次完整采样：从初始状态到结束的一整条轨迹 | 不是"一条 completion"，是**整个 episode** |
| **trace** | rollout 的记录：每轮的 token id、logprob、loss mask | 不是纯文本日志，是**能反传梯度的张量记录** |

### 最容易混的一对：harness vs environment

```
harness（壳）：决定"下一步做什么"
  · 把工具列表塞进上下文
  · 解析模型输出里的 tool_call
  · 把工具结果拼回上下文
  · 判断结束条件、压缩历史
        │ 调用工具
        ▼
environment（环境）：决定"做了会怎样"
  · 维护状态（文件系统、数据库、游戏盘面）
  · 执行动作、返回观测
  · 判定成败、给 reward
```

一句话：**harness 是"大脑的外壳"，environment 是"身体所在的世界"**。
在 TRL 里：`environment_factory` 给你 environment；`HarnessRolloutWorker` 给你 harness。

---

## 4. 全景图：一次 agent rollout 到底发生了什么

```
任务 prompt
  │
  ▼
actor（模型本身，就是要训练的那份权重）
  │   1 输出 tool_call
  ▼
harness（循环控制器：解析工具调用、执行、把结果拼回上下文）
  │   2 执行
  ▼
environment（有状态的世界：文件、shell、浏览器、数据库……）
  │   3 返回观测（stdout / 报错 / 页面内容）
  ▼
回到 actor 的上下文 ── 重复 1 2 3，直到模型说"我完成了"或步数用尽
  │
  ▼
4 整条轨迹 + 一个结果
  │
  ├─ 有标准答案？──→ SFT（抄轨迹） / 蒸馏（抄分布）
  └─ 只有分数？  ──→ GRPO（按分数改自己） / + 环境（按多轮结果改）
```

**关键认知**：后训练算法本身没变（还是交叉熵、还是策略梯度）。
变的是**"监督信号从哪来"**，以及**"一条样本有多长"**（从 1 个回答 → 几十轮交互）。

---

## 5. 监督信号的谱系

这是整份教程最重要的一张表。五步就是在沿着这张表往右下走。

| | 监督信号 | 密度 | 需要什么 | 学不到什么 |
|---|---|---|---|---|
| SFT | 每个 token 的**硬标签** | 最密（每 token 1 个） | 正确轨迹 | 不会纠错、不会探索 |
| 蒸馏 | 每个 token 的**概率分布** | 密（每 token 1 个分布） | 教师模型 | 不会超过教师 |
| GRPO | 每条回答**一个分数** | 稀（每序列 1 个） | 奖励函数 | 需要大量采样 |
| RL 环境 | 每次 **episode 一个结果** | 更稀（多轮 1 个） | 环境 + 验证器 | rollout 慢 |
| Harness | 真实 agent 跑完的**结果** | 最稀、最真实 | harness + 沙箱 | 工程复杂、成本高 |

> 一个直觉：**信号越稀，模型越要靠自己"试"；信号越密，模型越只是在"抄"。**
> 抄得快但不鲁棒，试得慢但能超越数据——这就是为什么真实流水线是 1 → 2 → 3/4 → 5 串起来的。

---

## 6. 环境准备

### 6.1 依赖

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install "trl>=1.16.0" transformers datasets accelerate
pip install trl[vllm]              # 想用 vLLM 加速生成（3/4/5 步强烈建议）
pip install openenv-core           # 第 4 步：环境
```

> 版本提示：第 4 步的 `environment_factory` 要求 `transformers>=5.2.0`；
> 第 5 步的异步 rollout 走 NCCL 权重同步，vLLM 版本别太旧。

### 6.2 硬件

| 步骤 | 最小可行 | 舒服 |
|---|---|---|
| 1 SFT | 1×16GB（0.5B + QLoRA） | 1×A100 |
| 2 蒸馏 | 1×24GB（学生 0.5B + 教师 1.5B） | 1×A100 |
| 3 GRPO | 1×24GB（0.5B，G=8） | 4–8×A100 |
| 4 环境 | 同 3，但要留 rollout 时间 | 8×A100 |
| 5 Harness | **≥2 卡**（vLLM 与 trainer 分卡） | 多机 + 沙箱配额 |

> 手上没有 24GB 以上的卡？本章第 **9～15 节**是完整的云端方案：怎么选机器、代码怎么送上去（git / rsync）、开机器后的初始化脚本、单卡与双卡的具体命令、四个云上专属的坑、以及一次走完大概花多少钱。

### 6.3 三条命令先跑通

```bash
# 1) 确认能加载 chat template 和工具 schema
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); print(t.chat_template[:200])"

# 2) 确认 TRL 装好
python -c "import trl; print(trl.__version__)"

# 3) 确认 GPU 可见
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

---

## 7. 本教程的贯穿示例

为了让五步能横向对比，我们始终用**同一个任务族**：一个会读文件、改文件、跑测试的**编程 agent**。

```
工具集（tools）：
  read_file(path: str) -> str          读文件
  write_file(path: str, content: str)  写文件
  run_tests() -> str                   跑测试，返回通过/失败

任务：修好 calculator.py 里失败的测试
```

| 步骤 | 这个任务怎么用 |
|---|---|
| 1 SFT | 一条"人类/强模型修好了"的完整 transcript |
| 2 蒸馏 | 只给任务描述，学生自己修，教师逐 token 打分 |
| 3 GRPO | 换成单轮可验证数学题（先学会奖励函数怎么写） |
| 4 环境 | 真给一个有文件系统的环境，多轮交互，测试通过才得分 |
| 5 Harness | 真跑 mini-swe-agent / claude-code，在沙箱里跑完整循环 |

---

## 8. 怎么用这份教程

1. **先读本文件**（你正在读），把六个词记住。
2. 进入第 1 章，**先看「输入格式」那张表和 JSONL 示例**——那是你实际要准备的东西。
3. 再看「输入输出」小节，理解 `labels` 里的 `-100` 是怎么来的。
4. 跑 `code/` 里的脚本，盯「产物与验收」里列的指标。
5. 每章最后的「练习」做完再进下一章。

---

[← 总目录](../README.md) · [下一节：SFT →](../01-监督微调-SFT/README.md)
