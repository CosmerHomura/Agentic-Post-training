# 02 · 蒸馏（Distillation）

[← 上一节：SFT](../01-监督微调-SFT/README.md) · [总目录](../README.md) · [下一节：GRPO →](../03-强化学习-GRPO/README.md)

---

## 1. 一句话定义

**不给"标准答案"，只给"教师的概率分布"——而且是在学生自己写出来的句子上给。**

SFT 是"抄正确答案"，蒸馏是"抄老师的思路"。差别在于：老师的思路在**每一个 token 上都有一整张概率表**，信息量比一个硬标签大得多。

---

## 2. 原文译文 · 2. Distillation

> 由一个教师模型来监督学生：监督可以来自**教师写出来的数据**，也可以来自**教师在学生自己那些 token 上的概率**。教师通常是一个更大的模型，也可以是"多个专家模型蒸馏进一个学生"（例如 MOPD），或者在自蒸馏里是"同一个模型 + 更长的上下文"。
>
> 这个信号是**密集的：每个 token 一个值**，所以训练所需的样本量比 RL 少。
>
> 反过来，学生会被教师的知识水平所限制；而在 on-policy 蒸馏里，**每一个训练步都必须跑一次教师**。RL 则用"只看结果的奖励"替代了教师。

### 原文推荐资源

| 类型 | 链接 |
|---|---|
| Watch | [Training Agents 2: distillation](https://www.youtube.com/watch?v=GZT_oNQnLV4)（[合集](https://huggingface.co/collections/sergiopaniego/training-agents-class-2-distillation-6a47a62b9f09867dccf61e1e)） |
| Read | [A brief history of distillation in AI](https://huggingface.co/blog/sergiopaniego/brief-history-of-distillation-in-ai) · [Distillation in 2026 (so far)](https://huggingface.co/blog/sergiopaniego/distillation-2026) |
| Docs | [Distillation Trainer](https://huggingface.co/docs/trl/distillation_trainer) |

---

## 3. 三种蒸馏，别混

| 类型 | 数据从哪来 | 监督是什么 | 在 TRL 里 |
|---|---|---|---|
| **离线蒸馏** | 教师先批量写好答案 | 硬标签（token id） | 其实就是 SFT（用教师产的数据） |
| **On-policy 蒸馏（GKD）** | **学生当场生成** | 教师在**学生生成的那串 token** 上的完整分布 | `DistillationTrainer` |
| **自蒸馏** | 学生生成（教师=同一模型+更多上下文） | 同上 | 同为 `DistillationTrainer`，教师换配置 |

本教程讲的是第二种，也是 TRL `DistillationTrainer` 实现的那一种（GKD，论文 2306.13649）。

> 为什么必须 on-policy？因为学生会犯错。离线蒸馏里学生的错误从未被教师"点评"过；
> on-policy 让学生在**自己会走的路径上**收到教师的纠正信号——这正是原文说的"用模型自己的输出训练"。

---

## 4. 训练信号从哪来

| 环节 | 内容 |
|---|---|
| **输入** | 只有 `prompt`（任务描述 + 工具列表） |
| **学生产出** | 自己生成的一条 completion（可能含工具调用） |
| **教师产出** | 在**同一串 token** 上，每个位置的完整词表分布 `p_T(·|prefix)` |
| **监督信号** | 学生分布 `p_S` 与教师分布 `p_T` 的**逐 token 散度** |
| **损失** | `L = Σ_t [ β·KL(p_T‖p_M) + (1-β)·KL(p_S‖p_M) ]`，`p_M = (1-β)p_S + β·p_T` |
| **产物** | 分布上贴近教师的模型（通常比 SFT 更"懂"细节） |

```
        prompt
          │
          ├──────────────▶ 学生：自回归生成 completion  y_1..y_T
          │                        │
          │                        ├──▶ 学生 forward → p_S(·|y_<t)   [T × V]
          │                        │
          └──────────────▶ 教师：对同一串 y 做 forward
                                   │
                                   └──▶ 教师 forward → p_T(·|y_<t)   [T × V]
                                             │
                                             ▼
                              JSD(p_S ‖ p_T) 逐 token 求，取平均 → loss
```

**关键点**：教师不是"重新写一遍"，而是**给学生写的那串 token 打分**。所以师生必须用同一个 tokenizer。

---

## 5. 输入格式

### 5.1 数据集：只要 prompt

| 字段 | 类型 | 含义 |
|---|---|---|
| `prompt` | `list[dict]` | 对话式消息列表，**没有** `completion` |

> 注意是**对话式**的 `prompt`（messages 列表），不是字符串。TRL 会套 chat template。
> 如果学生要调工具，可在数据集里加 `tools` 列（见文档 Agent Training 一节），这里先保持最小。

`data/distill_prompts.jsonl`：

```json
{"prompt": [{"role": "system", "content": "你是一个编程 agent。你可以读文件、写文件、跑测试。"},
            {"role": "user", "content": "calculator.py 里的测试失败了，请修好它。"}]}
```

一行就够——**答案不存在**，答案由学生当场写。

### 5.2 与 SFT 数据集的对比

| | SFT | 蒸馏（GKD） |
|---|---|---|
| 有 `completion`/`messages` 答案吗 | 有（完整正确轨迹） | **没有** |
| 一条数据训练几次 | 1 次（固定答案） | 每次都用新生成的 completion |
| 数据规模需求 | 大（要覆盖各种情况） | 小（prompt 少也能学） |
| 一条数据能反复用吗 | 可以但会过拟合 | **天然复用**，每次生成不同 |

---

## 6. 输入输出：三层张量

### 6.1 学生侧（与 SFT 相同的自回归前向）

```
input_ids = prompt tokens ⧺ completion tokens        [1, L]
labels    = -100 ... -100 ⧺ completion tokens        （mask 与 SFT 相同）
logits    = [1, L, V]  →  softmax(T)  →  p_S
```

### 6.2 教师侧（只对 completion 段做前向）

```
teacher_logits = [1, L, V]  →  softmax(T)  →  p_T
                 只取 completion 段的位置（prompt 段丢掉）
```

> 实现上 TRL 会**分块**计算（chunked），避免一次性物化 `[L, V]` 的大张量 ——
> 33k 词表 × 4k 序列 × bf16 就是 260MB/样本，不分块会 OOM。

### 6.3 损失长什么样（β 的三个取值）

设 `p_S` 学生分布、`p_T` 教师分布：

| β | 公式 | 直觉 | 什么时候用 |
|---|---|---|---|
| `0.0` | `D_KL[p_T ‖ p_S]` | **前向 KL**：教师有概率的地方学生也必须有 → 学生"铺得开" | 想让学生保留多样性 |
| `1.0`（默认） | `D_KL[p_S ‖ p_T]` | **反向 KL**：学生只学教师最确定的部分 → 学生"学得尖" | 想要一个更确定的学生 |
| `0.5` | 标准 JSD | 折中，数值稳定 | 通用默认，推荐先试 |

```
        β=0（前向 KL）                  β=1（反向 KL）
   学生必须覆盖教师的所有峰        学生只盯教师的最高峰
   ┌───┐   ┌───┐                    ┌───┐
   │   │   │   │                    │   │
   └───┴───┴───┘                    └───┘
   更"散"、更全                     更"尖"、更稳
```

> ⚠️ `DistillationConfig.beta` 和 `GRPOConfig.beta` **同名但含义完全不同**：
> 这里是"选哪种散度"，在 GRPO 里是"对参考模型的 KL 惩罚系数"。别抄错。

---

## 7. 代码与参数

完整脚本：[`code/train_distill.py`](code/train_distill.py)

```python
from datasets import load_dataset
from trl import DistillationConfig, DistillationTrainer

dataset = load_dataset("json", data_files="../data/distill_prompts.jsonl", split="train")

args = DistillationConfig(
    output_dir="./ckpt/distill-agent",
    beta=0.5,                    # 散度选择（0=前向KL, 1=反向KL, 0.5=JSD）
    temperature=1.0,             # 生成与损失共用；调高→教师分布更软
    max_completion_length=1024,  # 学生单条生成上限
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    bf16=True,
    gradient_checkpointing=True,
)

trainer = DistillationTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
    args=args,
    train_dataset=dataset,
)
trainer.train()
```

### 关键参数

| 参数 | 作用 | 默认 | 建议 |
|---|---|---|---|
| `beta` | 散度选择器（见上表） | `1.0` | 先 `0.5` |
| `temperature` | 同时给生成与损失用 | `1.0` | 1.0~2.0；越高信息越多也越抖 |
| `max_completion_length` | 学生生成上限 | `512` | 多轮工具调用要调大 |
| `teacher_model` | 教师 | 必填 | 比学生大 1 个量级即可 |
| `use_vllm` / `vllm_mode` | 用 vLLM 生成学生 rollout | `False` / `colocate` | 规模上来后必开 |
| `max_tool_calling_iterations` | 允许的最多工具轮数 | — | 多轮 agent 时设 |
| `remove_unused_columns` | 是否删掉数据集多余列 | `False` | 蒸馏里保持 `False` |

> 小细节：`DistillationConfig` 的默认值和 `TrainingArguments` 不一样——
> `logging_steps=10`、`gradient_checkpointing=True`、未设 `fp16` 时 `bf16=True`、`lr=1e-6`。

---

## 8. 产物与验收

### 8.1 日志指标

| 指标 | 含义 | 健康表现 |
|---|---|---|
| `loss` | JSD/KL 散度均值 | 下降；但**不会降到 0**（学生永远追不上教师） |
| `num_tokens` | 累计 token | 估算吞吐 |
| `step_time` | 每步耗时 | 明显高于 SFT（多一次教师前向） |
| `completions/mean_length` | 学生生成平均长度 | 别一路涨到上限（说明不会停） |
| `completions/mean_terminated_length` | 正常以 EOS 结束的平均长度 | 与上面接近才健康 |
| `completions/clipped_ratio` | 被 `max_completion_length` 截断的比例 | 应接近 0 |
| `completions/min_length` / `max_length` | 长短极值 | 看是否有退化 |
| `entropy` | 学生输出熵 | 大幅下降=在变确定；掉到 0=塌缩 |
| `tools/call_frequency` | 每条 completion 平均工具调用数 | agent 场景的核心观察项 |
| `tools/failure_frequency` | 工具调用失败频率 | 应该随训练下降 |

### 8.2 判断"蒸馏成功"的三个问题

```
□ 学生是否学会了教师那种「先看再改」的顺序？（对比 SFT 基线）
□ 学生生成长度是否收敛，而不是越写越长/越写越短？
□ 在教师也做不对的题上，学生是否明显更差？（这会暴露"教师上限"）
```

### 8.3 产物

- `./ckpt/distill-agent/`：分布上更贴近教师的模型
- 一个可复用的**教师服务**（后面做 RL 时可以拿它当 reward model 的替代）

---

## 9. 常见坑

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | 师生 tokenizer 不同 | token 下标错位，loss 乱飞 | 强制同一 tokenizer |
| 2 | 把 `beta` 当成 GRPO 的 KL 惩罚 | 设成 0.04，结果学生塌缩 | 记住：这里是**散度选择器** |
| 3 | `temperature` 只改生成不改损失（或反过来） | 教师分布软硬不一致 | TRL 里它是共用的，别手动覆盖 |
| 4 | 序列太长 | OOM | 减小 `max_completion_length`；依赖分块实现 |
| 5 | 教师也答不对 | 学生天花板低（原文明确） | 换更强教师，或转 RL |
| 6 | 每步都跑教师 → 慢 | 吞吐只有 SFT 的 1/2~1/3 | 开 `use_vllm`，教师用更小模型 |
| 7 | 数据里塞了 completion | 变成离线蒸馏（等价 SFT） | 确认数据集只有 `prompt` |

---

## 10. 练习与自测

1. **β 扫描**：固定其他配置，跑 `beta=0.0 / 0.5 / 1.0` 三组，各训 100 步，比较 `entropy` 与生成多样性（同一 prompt 采样 8 次看差异）。
2. **教师上限实验**：故意选一个**比学生弱**的教师，观察 loss 是否照样下降、但下游任务变差——亲手感受"学生被教师限制"。
3. **对比 SFT**：用第 1 章的数据先做 SFT，再用同一批 prompt 做蒸馏，在 10 个留出任务上比成功率。

---

## 11. 一句话带走

> 蒸馏把"一个硬标签"换成"一整张概率表"，信号变密、样本变省；
> 但它**永远以教师为天花板**，而且要不停付"跑教师"的钱。下一章的 RL，就是用"一个只看结果的分数"把教师替换掉。

---

[← 上一节：SFT](../01-监督微调-SFT/README.md) · [总目录](../README.md) · [下一节：GRPO →](../03-强化学习-GRPO/README.md)
