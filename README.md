# Agent 后训练学习教程

> 对照 Hugging Face 官方课程页 **《A curriculum for post-training agents》**
> 原文：<https://huggingface.co/blog/huggingface/train-agents>
> 本教程：全文中文翻译 + 每一步的**数据格式**、**输入输出**、**可运行代码骨架**、**踩坑清单**
> 整理日期：2026-10-28 ｜ 对应文档版本：TRL v1.16.0

---

## 0. 这份教程是什么

Hugging Face 那篇课程页是一张"**学习地图**"：它把 agent 后训练拆成 5 步，每步给你「一张方法图 + 一场直播 + 一篇博客 + 官方文档」。
但它本身**不告诉你每一步的数据长什么样、喂给模型的是什么、模型吐出来的是什么**——这些细节散落在 TRL 文档各个角落。

这份教程补上这一块：

| 你会得到 | 在哪 |
|---|---|
| 原文的完整中文翻译（逐节对应） | 各章 README 的「原文译文」小节 |
| 每一步的**数据格式**（JSONL 逐字段） | 各章 `data/` 示例文件 + README 表格 |
| 每一步的**模型输入输出**（token / mask 视角） | 各章 README 的「输入输出」小节 |
| 可运行代码骨架（改路径就能跑） | 各章 `code/` 目录 |
| 训练成败的判断依据（看哪些指标） | 各章 README 的「产物与验收」小节 |
| 目录导航、学习路线、资源总表 | 本文件 |

**网页版**：仓库里的 `index.html` 是整份教程的单文件网页（左边目录、小节可打勾、链接可加书签、自动记进度）。**下载下来双击就能看**，不需要服务器、不需要联网。打勾存在你自己浏览器的 localStorage 里，所以谁下载谁记自己的，互不影响；想清空就点页面左下角「重置进度」。

**一句话主线**：后训练就是"换监督信号"。
从「抄一遍正确答案」（SFT）→「抄老师的概率分布」（蒸馏）→「按评分改自己」（RL）→「在环境里按结果改自己」（RL 环境）→「在真实 agent 里按结果改自己」（Harness）。
**监督信号越来越稀疏，但越来越贴近你真正想要的能力。**

---

## 1. 原文译文 · 引言

> 后训练（post-training）指的是预训练之后的那些训练阶段，例如监督微调、蒸馏和强化学习。这些阶段把一个预训练模型变成会遵循指令、会使用工具、能解决特定任务的模型。
>
> 当模型作为 agent 工作时，它要在很多轮里行动：调用工具、读取结果、决定下一步做什么。**Agentic 后训练**就是把这些阶段用在 agent 上——于是训练必须考虑整条多轮交互的序列，而不是单个回答。

> 本页组织为五个步骤，前面有一段导言。每一步都包含：方法示意图、一场直播或幻灯片、一篇解释思路的博客，以及相关文档。部分步骤还附了用该方法构建的示例项目。下面的表格总结了每一步：模型从什么信号中学习、用哪个 TRL trainer、你需要准备什么。

> 这些步骤的顺序沿袭直播系列 **Training Agents**（Hugging Face 的 YouTube 频道）。每一期讲一种后训练方法，并始终以"训练一个编程 agent"作为贯穿全程的示例。播放列表 [Post-training Agents](https://www.youtube.com/playlist?list=PLo2EIpI_JMQvQZm-kVlz4wY1vWF0LBcf5) 包含四期节目加两场相关 workshop。

---

## 2. 五步一图看懂

```
   同一个模型，五种监督信号
   ══════════════════════════════════════════════════════════════════════

   (1) SFT          数据：(prompt, 完整正确轨迹)      信号：每个 token 的硬标签
      模仿          ─────────────────────────────▶  loss = 交叉熵

   (2) 蒸馏         数据：(prompt)  学生自己生成      信号：教师每个 token 的分布
      对齐分布      ─────────────────────────────▶  loss = JSD(p_学生 ‖ p_教师)

   (3) RL/GRPO      数据：(prompt, 可验证答案)        信号：整条回答一个分数
      试错优化      ─────────────────────────────▶  loss = 组内相对优势 × 策略梯度

   (4) RL 环境      数据：(任务) + 有状态环境        信号：多轮交互后的最终结果
      行动学习      ─────────────────────────────▶  同上，但轨迹由环境决定长度

   (5) Harness      数据：(任务) + 真实 agent + 沙箱  信号：真实 agent 跑完的结果
      规模化        ─────────────────────────────▶  同上，但生成与训练解耦并行

        监督密度：  密集(每token) ────────────────▶ 稀疏(每个 episode 一个数)
        真实程度：  离线数据   ────────────────▶ 生产环境里的同一个 agent
```

### 原文「Steps at a glance」表格（译文）

| 步骤 | 模型从什么学习 | TRL trainer | 你需要准备 |
|---|---|---|---|
| 1. SFT | 来自更强模型的 agent 轨迹 | `SFTTrainer` | 一个 agent 轨迹数据集 |
| 2. 蒸馏 | 教师的 token 或概率 | `DistillationTrainer` | prompts + 一个教师模型 |
| 3. RL | 每个回答的得分 | `GRPOTrainer` | prompts + 一个奖励函数 |
| 4. RL 环境 | 在环境中行动的结果 | `GRPOTrainer` | 任务 + 一个环境（OpenEnv） |
| 5. Harness 与规模化 | 真实 agent 的完整运行 | `AsyncGRPOTrainer` | 任务 + 环境 + 一个 agent harness |

### 换一种说法：每一步的输入 / 输出 / 产物

| 步骤 | 你喂给 trainer 的 | 模型（或环境）产出的 | 训练后你拿到 |
|---|---|---|---|
| 1 SFT | `messages`（含 `tool_calls`）| 每个 token 的预测 → 交叉熵 | 会模仿轨迹的模型 |
| 2 蒸馏 | 只有 `prompt` | 学生自己写的回答 + 师生两组分布 | 分布上贴近教师的模型 |
| 3 GRPO | `prompt` + `ground_truth` | G 条回答 + G 个分数 | 会得高分的模型 |
| 4 环境 | 任务（可由环境自采样）| 多轮 obs/action 序列 + 最终奖励 | 会"用工具完成任务"的模型 |
| 5 Harness | 任务 + agent 配置 | 真实 agent 的完整 transcript + 验证器打分 | 在真实 harness 里变强的模型 |

---

## 3. 目录导航

```
Agentic Post-training/
├── README.md                       ← 你在这里（总目录 / 学习路线 / 资源总表）
├── 00-开始之前/
│   ├── README.md                   术语、全景图、环境准备、如何读这份教程
│   └── cloud-setup.md              只有云 GPU 时怎么跑：选机器 / 传输 / 命令 / 账单
├── .gitignore · .gitattributes     不提交 ckpt 与缓存；统一 LF
├── 01-监督微调-SFT/
│   ├── README.md                   译文 + 格式 + 输入输出 + 损失手算 + 代码 + 踩坑
│   ├── data/sft_agent_trace.jsonl  解剖样本（2 条完整轨迹，训练请用 trl-lib/Capybara）
│   └── code/train_sft.py           四阶段打印 + 训练 + 前后评测
├── 02-蒸馏/
│   ├── README.md
│   ├── data/distill_prompts.jsonl  解剖样本（3 条 prompt，训练请用 trl-lib/ultrafeedback-prompt）
│   └── code/train_distill.py       分布对比 + KL 手算 + 训练 + 前后 KL 评测
├── 03-强化学习-GRPO/
│   ├── README.md
│   ├── data/grpo_prompts.jsonl     解剖样本（3 道题，训练请用 trl-lib/DeepMath-103K）
│   └── code/train_grpo.py          采样打分 + 优势计算 + 损失手算 + 训练 + pass@1 评测
├── 04-RL环境/
│   ├── README.md
│   └── code/{env_factory.py, train_grpo_env.py}
├── 05-Harness与规模化/
│   ├── README.md
│   └── code/{train_async_grpo.py, harbor_rollout.py}
└── 99-附录-资源与术语/
    └── README.md                   术语表、原文全部链接、常见问题
```

### 按顺序读

| # | 章节 | 核心问题 | 预计 |
|---|---|---|---|
| 0 | [开始之前](00-开始之前/README.md) | agent / harness / scaffold / environment 到底指什么？ | 20 min |
| 1 | [监督微调 SFT](01-监督微调-SFT/README.md) | 一条 agent 轨迹长什么样？怎么只对 assistant 的 token 算 loss？ | 60 min |
| 2 | [蒸馏](02-蒸馏/README.md) | 没有"标准答案"时，怎么用教师的概率分布当监督？ | 45 min |
| 3 | [强化学习 GRPO](03-强化学习-GRPO/README.md) | 只有一个分数时，梯度怎么算？奖励函数怎么写？ | 75 min |
| 4 | [RL 环境](04-RL环境/README.md) | 多轮交互、工具、状态，怎么接进 trainer？ | 60 min |
| 5 | [Harness 与规模化](05-Harness与规模化/README.md) | 真实 agent 自己掌控循环时，怎么训练？怎么并行？ | 60 min |
| 附 | [资源与术语](99-附录-资源与术语/README.md) | 原文全部链接、名词对照、FAQ | 随时查 |

### 三条学习路线

- **速通（半天）**：读本文件 → 每章只读「原文译文」+「输入输出」两张表 → 跳过代码。
- **动手（一周）**：按 1→2→3→4→5 顺序，每章跑一遍 `code/` 里的脚本（用小模型 `Qwen3-0.6B`）。
- **深挖（两周）**：每章读完官方文档链接 → 把你自己的任务改造成对应格式 → 对比五步在同一任务上的效果。

---

## 4. 每个文件夹的统一结构

每章 README 都是同一套骨架，方便你横向对比：

```
1. 顶部导航        ← 上一节 / 总目录 / 下一节
2. 一句话定义      ← 这步在整条流水线里的位置
3. 原文译文        ← 该节完整翻译（含 Watch/Read/Docs 链接）
4. 训练信号从哪来  ← 一张表：输入 → 产出 → 监督 → 损失
5. 输入格式        ← JSONL 逐字段表 + 真实示例
6. 输入输出        ← token / mask / logits 视角，带示意图
7. 代码与参数      ← 最小可运行脚本 + 关键参数表
8. 产物与验收      ← 该看哪些日志指标、怎么判断训成功了
9. 常见坑          ← 我们踩过的（和文档里明写的）
10. 练习与自测     ← 3 个动手题
```

---

## 5. 环境与依赖

```bash
# 最小可用（第 1–3 步，单卡即可）
pip install "trl>=1.16.0" transformers datasets accelerate

# 第 4 步（环境）
pip install openenv-core        # 或直接用 TRL 的 environment_factory，不装也能跑
pip install trl[vllm]           # 用 vLLM 加速生成

# 第 5 步（异步 / harness）
#   - 需要至少 2 张 GPU：一张跑 vLLM server，一张跑 trainer
#   - 黑盒 harness 还需要 OpenEnv Harbor server + 沙箱凭证（如 E2B_API_KEY）
```

硬件参考：`0.5B–0.6B` 模型单卡 16GB 可跑 SFT / 蒸馏 / GRPO 小规模实验；`4B` 以上建议 A100 级别；第 5 步按文档说明需要把 vLLM 与训练分卡。

---

## 6. 名词速查（详见 [附录](99-附录-资源与术语/README.md)）

| 词 | 一句话 |
|---|---|
| **agent** | 能多轮调用工具、根据结果决定下一步的模型 |
| **harness**（脚手架/外部壳） | 驱动 agent 的那套程序：规划、工具集、上下文管理、停止条件 |
| **scaffold** | 与 harness 近义，常指"模型之外的工程结构" |
| **environment** | 有状态的世界：能执行动作、记住变化、判定结果 |
| **rollout** | 一次完整的采样过程（一条轨迹 / 一个 episode） |
| **trace** | rollout 的记录：每轮的 prompt、completion、token、logprob、mask |
| **on-policy** | 用"当前模型自己生成的"数据来训练 |
| **reward** | 给整条回答/整次 rollout 的一个标量分数 |

---

## 7. 原文资源总表

### 7.1 通用

| 类型 | 资源 |
|---|---|
| 课程页 | <https://huggingface.co/blog/huggingface/train-agents> |
| 播放列表 | [Post-training Agents](https://www.youtube.com/playlist?list=PLo2EIpI_JMQvQZm-kVlz4wY1vWF0LBcf5)（四期正课 + 两场 workshop） |
| 术语博客 | [Harness, Scaffold, and the AI Agent Terms Worth Getting Right](https://huggingface.co/blog/agent-glossary) |
| 总览幻灯片 | [Train the agent, not just the model](https://huggingface.co/spaces/sergiopaniego/train-the-agent-slides)（20 页） |
| 入门课 | [LLM Course](https://huggingface.co/learn/llm-course) |

### 7.2 逐步资源（对应原文各节的 Watch / Read / Projects / Docs）

| 步骤 | 直播 / 幻灯片 | 阅读 | 项目 | 文档 |
|---|---|---|---|---|
| **1 SFT** | [Training Agents 1](https://www.youtube.com/watch?v=rNgUoH7Wbv8)（[直播代码](https://huggingface.co/burtenshaw/gemma4-pi-mono-youtube-livestream-1-scripts)） | [SFT from scratch](https://x.com/ben_burtenshaw/status/2067615361428545566) | — | [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) |
| **2 蒸馏** | [Training Agents 2](https://www.youtube.com/watch?v=GZT_oNQnLV4)（[合集](https://huggingface.co/collections/sergiopaniego/training-agents-class-2-distillation-6a47a62b9f09867dccf61e1e)） | [蒸馏简史](https://huggingface.co/blog/sergiopaniego/brief-history-of-distillation-in-ai) · [Distillation in 2026](https://huggingface.co/blog/sergiopaniego/distillation-2026) | — | [Distillation Trainer](https://huggingface.co/docs/trl/distillation_trainer) |
| **3 GRPO** | [Training Agents 3](https://www.youtube.com/watch?v=ztdTed5egrM)（[合集](https://huggingface.co/collections/sergiopaniego/training-agents-class-3-reinforcement-learning-6ad6aa71050342629dd0e3ec)） | [Outcome-based RL 2026](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) | [350M 模型 100 步 GRPO 学结构化输出](https://huggingface.co/blog/grpo-with-trl-ifstruct) | [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) |
| **4 环境** | [Training Agents 4](https://www.youtube.com/watch?v=nJV3yUuz6DU)（[合集](https://huggingface.co/collections/sergiopaniego/training-agents-class-4-rl-environments-6aa27543918888a0f2b5312c)） · [RL for Agents Workshop](https://www.youtube.com/watch?v=cixmqTsi2A4) | [RL 环境为什么现在可行](https://huggingface.co/blog/sergiopaniego/rl-environments-story) | [用代码画水彩](https://huggingface.co/blog/train-to-paint-with-code) · [CARLA 自动驾驶环境](https://huggingface.co/blog/sergiopaniego/bringing-carla-to-openenv-trl) | [TRL + OpenEnv](https://huggingface.co/docs/trl/openenv) · [OpenEnv](https://huggingface.co/docs/openenv) |
| **5 Harness** | [Multi-harness RL 幻灯片](https://huggingface.co/spaces/FineEnvs/multi-harness-rl-slides) | [Multi-harness RL 指南](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) · [每次 rollout 一个沙箱](https://huggingface.co/blog/sergiopaniego/rl-environments-2026) · [Keep the Tokens Flowing](https://huggingface.co/blog/async-rl-training-landscape) | — | [Async GRPO Trainer](https://huggingface.co/docs/trl/async_grpo_trainer) · [OpenEnv harnesses](https://huggingface.co/docs/openenv/tutorials/harnesses) |

> 完整清单（含合集、配图、原文逐条链接）见 [附录 B · 资源总表](99-附录-资源与术语/README.md)。

---

## 8. 来源、版权与引用

这份教程是对 Hugging Face 博客 **《A curriculum for post-training agents》**（<https://huggingface.co/blog/huggingface/train-agents>）的学习笔记：原文的中文译文、逐节对照在 [附录 A](99-附录-资源与术语/README.md)，原文的全部外链在 [附录 B](99-附录-资源与术语/README.md)。

- **原文版权**归 Hugging Face 及其作者所有。译文只用于学习与对照，不是官方译本。
- 本仓库**文字部分**（中文讲解、表格、示意图、`cloud-setup.md`）按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 共享：可以转载、改写、商用，注明出处即可。
- **代码部分**（各章 `code/*.py`、`_site/` 构建脚本）按 [MIT](LICENSE) 授权。
- 欢迎 issue / PR 纠错。发现译错的段落，直接改对应 `README.md`，然后 `python _site\build.py` 重新生成网页再提交。

> 别人照这份仓库学，最好的引用方式是给个链接：`https://github.com/CosmerHomura/Agentic-Post-training`。

---

**开始读 →** [00 · 开始之前](00-开始之前/README.md)
