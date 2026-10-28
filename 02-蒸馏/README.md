# 02 · 蒸馏（Distillation）

[← 上一节：监督微调 SFT](../01-监督微调-SFT/README.md) · [总目录](../README.md) · [下一节：强化学习 GRPO →](../03-强化学习-GRPO/README.md)

---

## 1. 一句话定义

**没有标准答案：学生自己写一版，教师逐 token 告诉它「换了我，每个词该有多大概率」，学生去对齐这个概率分布。**

SFT 给的是**一个**答案（硬标签）；蒸馏给的是**一整张概率表**（软标签）。所以同样的数据量，蒸馏的信号密度高得多。

---

## 2. 原文译文 · 2. Distillation

> 蒸馏（distillation）用教师的 token 或概率来训练学生模型，而不是用硬标签。因为教师给了每个 token 一个值（或者说一整条分布），每个样本携带的信号比 SFT 更多，所以蒸馏往往需要更少的数据。
>
> 但它有三个代价：学生的上限被教师封住；每一步训练都要额外跑一次教师（更慢、更占显存）；而且必须是 **on-policy** 的——学生自己生成，教师在这些 token 上打分，否则学生在自己会走的路径上收不到纠正。

### 原文推荐资源

| 类型 | 链接 |
|---|---|
| Watch | [Training Agents 2: distillation](https://www.youtube.com/watch?v=GZT_oNQnLV4)（[代码合集](https://huggingface.co/collections/merve/training-agents-6a47a62b9f09867dccf61e1e)） |
| Read | [A brief history of distillation in AI](https://huggingface.co/blog/distillation-2026)、[Distillation](https://huggingface.co/blog/distillation-2026) |
| Docs | [Distillation Trainer](https://huggingface.co/docs/trl/distillation_trainer) |

---

## 3. 三种蒸馏，别混

| 类型 | 数据从哪来 | 监督是什么 | 在 TRL 里 |
|---|---|---|---|
| **离线蒸馏** | 教师先批量写好答案 | 硬标签（token id） | 其实就是 SFT（用教师产的数据） |
| **On-policy 蒸馏（GKD）** | **学生当场生成** | 教师在**学生生成的那串 token** 上的完整分布 | `DistillationTrainer` |
| **自蒸馏** | 学生生成（教师=同一模型+更多上下文） | 同上 | 同为 `DistillationTrainer`，教师换配置 |

本教程讲第二种，也是 TRL 实现的那一种（GKD，论文 2306.13649）。

---

## 4. 训练信号从哪来

| 环节 | 内容 |
|---|---|
| **输入** | 只有 `prompt`（任务描述） |
| **学生产出** | 自己生成的一条 completion |
| **教师产出** | 在**同一串 token** 上，每个位置的完整词表分布 `p_T(·|prefix)` |
| **监督信号** | 学生分布 `p_S` 与教师分布 `p_T` 的**逐 token 散度** |
| **损失** | `L = β·KL(p_T‖p_M) + (1-β)·KL(p_S‖p_M)`，`p_M = (1-β)p_S + β·p_T` |
| **产物** | 分布上贴近教师的模型 |

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
                             逐 token 散度 → 取平均 → loss
```

**关键点**：教师不是"重新写一遍"，而是**给学生写的那串 token 打分**。所以师生必须共用一个 tokenizer。

---

## 5. 全链路实拍：蒸馏一条样本，发生五件事

```
JSONL（只有 prompt）
   ──chat template (add_generation_prompt=True)──►  一段以「<|im_start|>assistant\n」结尾的文本
   ──tokenize──►  prompt 的 token id
   ──学生 generate──►  completion（on-policy 的那一步）
   ──学生/教师各 forward 一次──►  每个位置上两张概率表 ──► 散度 ──► loss
```

下面每段都是 `python train_distill.py --inspect` / `python train_distill.py` 会打印的内容。

> **说明**：带具体数字的段落是**预期的示意值**——脚本打印的字段与结构就是这些，数字随模型、采样和卡而变。
> 你跑完把 `runs/distill/report.md` 发我，我会把这里全部换成你的实测输出。

### 5.1 阶段 1：读 JSONL

```
  文件：.../02-蒸馏/data/distill_prompts.jsonl
  行数：3

  ── 第 1 条  (2 条消息) ──
    [system] 你是一个编程 agent。你可以读文件、写文件、跑测试。
    [user] calculator.py 里的测试失败了，请修好它。测试文件是 test_calculator.py。
```

**没有 `completion`、没有 `answer`、没有 `ground_truth`。** 这是蒸馏与 SFT 最根本的区别：

| | SFT | 蒸馏 |
|---|---|---|
| 数据里有答案吗 | 有（完整正确轨迹） | **没有** |
| 一条数据训练几次 | 1 次（同一份答案） | 每次都用新生成的 completion |
| 数据量需求 | 大 | 小（prompt 少也能学） |

### 5.2 阶段 2：套模板（结尾留白给模型）

`add_generation_prompt=True` 渲染出来是（`▸` 标出角色分隔符）：

```
▸ <|im_start|>system
  你是一个编程 agent。你可以读文件、写文件、跑测试。<|im_end|>
▸ <|im_start|>user
  calculator.py 里的测试失败了，请修好它。测试文件是 test_calculator.py。<|im_end|>
▸ <|im_start|>assistant
```

注意最后那个 **`▸ <|im_start|>assistant` 后面是空的**——它在等学生往下写。这个"留白位"就是模型的起点。

> 对照 SFT：SFT 的数据里 assistant 后面**已经写满了**；蒸馏的数据里 assistant 后面**是空的**。

### 5.3 阶段 3：tokenize（只有 prompt）

```
  prompt token 数：58

   idx  token_id  token
  ----------------------------
     0    151644  <|im_start|>
     1      8948  system
     2       198  \n
     3     56568  你
     ...
    55    151645  <|im_end|>
    56    151644  <|im_start|>
    57     77091  assistant
```

到这里为止，和 SFT 的输入完全一样。**区别从下一步开始。**

### 5.4 阶段 4：学生自己生成 completion

```
  学生：Qwen/Qwen2.5-0.5B-Instruct
  采样：temperature=1.0  max_new_tokens=160

  生成 137 个 token：
  --------------------------------------------------------------------------
  先看代码和测试。
  <tool_call>
  {"name": "read_file", "arguments": {"path": "calculator.py"}}
  </tool_call>
  ...
  --------------------------------------------------------------------------
```

这段文字**没有标准答案可以对照**：它可能对、可能错、可能中途跑偏。教师不改它，只是告诉它"换了我，每个位置会怎么分布"。

于是训练样本变成 `prompt_ids + gen_ids`（58 + 137 = 195 个 token），**loss 只算后 137 个位置**——和 SFT 的掩码逻辑一模一样，只不过掩码不是"数据规定的"，而是"学生自己写出来的那段"。

### 5.5 阶段 5：教师 vs 学生，逐 token 比分布

两个模型吃同一串 195 个 token，各输出 `[195, 151936]` 的 logits。取 completion 段的每个位置：

```
  i   学生写的词    p_学生   p_教师   教师首选      p_首选
  --------------------------------------------------------------
  0   先            0.412    0.637    先            0.637
  1   看            0.588    0.771    看            0.771
  2   代码          0.310    0.502    代码          0.502
  3   和            0.244    0.398    和            0.398
  ...
  12  {"            0.083    0.245    <tool_call>   0.551   ← 分歧最大的地方
```

最后一行是典型现象：**学生想直接写 JSON，教师想先写 `<tool_call>` 标签**。这个 token 的分布差会变成一份更大的梯度，把学生往教师那边拽。

把每个位置的散度平均起来，就是这一步的 loss：

```
  平均前向 KL  D_KL(p_T‖p_S) = 0.1832
  β = 0.5 时的蒸馏损失        = 0.0911
```

> 训练时 `DistillationTrainer` 每步都做这件事，且是**分块**算的：
> 一次性物化 `[seq, 151936]` 的 float32 是 195×151936×4 ≈ 118MB，序列一长就 OOM，
> 所以它按 chunk 算 KL，再累加。

### 5.6 一个训练步里的张量（形状速查）

```
  input_ids       [1, 195]              # prompt(58) + completion(137)
  completion_mask [1, 195]              # 前 58 个 0，后 137 个 1
  student_logits  [1, 195, 151936]      # 学生前向
  teacher_logits  [1, 195, 151936]      # 教师前向（多一份显存）
  loss            scalar                # 只在 mask=1 的位置上平均
```

**显存账**：比 SFT 多一整个教师的权重 + 一次教师前向的激活。0.5B 学生 + 1.5B 教师，24GB 卡上 `max_completion_length=1024` 很宽松。

---

## 6. 跑起来

脚本：[`code/train_distill.py`](code/train_distill.py)

```bash
cd 02-蒸馏/code

# ① 只跑数据链路（只要 tokenizer，不需要 GPU）
python train_distill.py --inspect

# ② 完整流程：学生生成 → 分布对比 → 训练 → 损失图 → 前后对比 → 写报告
python train_distill.py

# 常用变体
python train_distill.py --max-steps 20 --gen-tokens 96   # 冒烟
python train_distill.py --beta 0.0                       # 前向 KL
python train_distill.py --temperature 1.5                # 更软的教师分布
```

依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "trl>=1.16.0" transformers datasets accelerate
```

| 产物 | 内容 |
|---|---|
| `runs/distill/report.md` | **把这份发我**：分布表、loss 表、损失图、前后对比 |
| `runs/distill/tokens.txt` | 全量 token 表 |
| `runs/distill/loss.csv` | 每步 loss |
| `ckpt/distill-agent/` | 蒸馏后的学生 |

### 6.1 训练时终端长这样

```
  学生参数：494M
  教师参数：1544M

  step   1  loss 0.4213  lr 3.3e-07  grad 2.881  ent 1.42  len 118  clip 0.00
  step   2  loss 0.3702  lr 6.7e-07  grad 2.104  ent 1.35  len 124  clip 0.00
  step   3  loss 0.3511  lr 1.0e-06  grad 1.933  ent 1.31  len 121  clip 0.00
```

<!-- 待回填：真实 loss 表 + 损失图（跑完把 report.md 发我） -->

### 6.2 损失图

```
  loss 走势 (min 0.351 / max 0.421)
   0.421 |●
   0.404 | ●
   0.386 |  ●
   0.369 |   ●●
   0.351 |     ●●●
         +--------------------------------------
          step 1 → 6
```

<!-- 待回填：真实损失图 -->

### 6.3 训练前后对比

```
  ── 用例 1 ──
  prompt：calculator.py 里的测试失败了，请修好它。测试文件是 test_calculator.py。
  蒸馏前（基座学生）：……
  蒸馏后：……
```

<!-- 待回填：report.md 里的 before/after 两段 -->

判断蒸馏有没有起效，不要只看 loss（它永远降不到 0），看这三个：

| 看什么 | 训练前 | 训练后 |
|---|---|---|
| 有没有按教师的**结构**写（`<tool_call>`） | 直接写 JSON / 写自然语言 | 出现结构化标签 |
| 会不会**先读再改** | 一上来就猜内容 | 先 `read_file` |
| 生成长度 | 要么一句话结束、要么跑到上限 | 落在合理区间、能正常收尾 |

---

## 7. 参数速查

| 参数 | 作用 | 默认 | 建议 |
|---|---|---|---|
| `beta` | 散度选择器（见下表） | `1.0` | 先 `0.5` |
| `temperature` | **同时**给生成与损失用 | `1.0` | 1.0~2.0；越高教师分布越软、信息越多也越抖 |
| `max_completion_length` | 学生单条生成上限 | `512` | 多轮工具调用要调大；本脚本默认 160 |
| `teacher_model` | 教师 | 必填 | 比学生大 1 个量级即可 |
| `use_vllm` / `vllm_mode` | 用 vLLM 生成学生 rollout | `False` / `colocate` | 规模上来后必开 |
| `max_tool_calling_iterations` | 允许的最多工具轮数 | — | 多轮 agent 时设 |
| `remove_unused_columns` | 是否删掉数据集多余列 | `False` | 蒸馏里保持 `False` |

### β 的三个取值

| β | 公式 | 直觉 | 什么时候用 |
|---|---|---|---|
| `0.0` | `D_KL[p_T ‖ p_S]` | **前向 KL**：教师有概率处学生也必须有 → 学生"铺得开" | 想保留多样性 |
| `1.0`（TRL 默认） | `D_KL[p_S ‖ p_T]` | **反向 KL**：学生只学教师最确定的部分 → "学得尖" | 想要确定的学生 |
| `0.5`（脚本默认） | 标准 JSD | 折中、数值稳 | 通用首选 |

> ⚠️ `DistillationConfig.beta` 和 `GRPOConfig.beta` **同名但完全不同**：
> 这里是"用哪种散度"，GRPO 里是"对参考模型的 KL 惩罚系数"。别抄错。

> 小细节：`DistillationConfig` 的默认值与 `TrainingArguments` 不同——
> `logging_steps=10`、`gradient_checkpointing=True`、未设 `fp16` 时 `bf16=True`、`lr=1e-6`。

---

## 8. 产物与验收

### 8.1 日志指标

| 指标 | 含义 | 健康表现 |
|---|---|---|
| `loss` | 散度均值 | 下降，但**不会降到 0**（学生追不上教师） |
| `num_tokens` | 累计 token | 估吞吐 |
| `step_time` | 每步耗时 | 明显高于 SFT（多一次教师前向） |
| `completions/mean_length` | 学生平均生成长度 | 别一路顶到上限（说明不会停） |
| `completions/mean_terminated_length` | 正常以 EOS 结束的平均长度 | 与上面接近才健康 |
| `completions/clipped_ratio` | 被截断的比例 | 长期 >0.2 就调大 `max_completion_length` |
| `tools/call_frequency` | 平均每次生成调用几次工具 | 从 0 涨起来说明学会了用工具 |
| `tools/failure_frequency` | 工具调用失败率 | 应下降 |
| `entropy` | 输出分布熵 | 缓降；掉到 0 附近=塌缩 |

### 8.2 判断"蒸馏成功"的三个问题

1. 学生生成的**结构**像教师了吗（`<tool_call>`、`<think>`、收尾）？
2. 同样的 prompt，学生的答案**稳定性**提高了吗（多采几次看方差）？
3. 学生在**自己容易犯错的那些位置**上，`p_教师 - p_学生` 缩小了吗？（看 §5.5 那张表前后对比）

### 8.3 产物

- `ckpt/distill-agent/`：可以做 RL 的起点（通常比 SFT 版更"会写"）
- `runs/distill/report.md`：这次蒸馏的证据

---

## 9. 常见坑

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | 师生 tokenizer 不一致 | 蒸馏完全学不动（token 下标错位） | 用同一个 tokenizer / 同一个词表 |
| 2 | 把 `beta` 当成 GRPO 的 KL 系数 | 调了没反应，或学生塌缩 | 这里 beta 是"选散度"，不是惩罚 |
| 3 | `max_completion_length` 太小 | `clipped_ratio` 高，学生学不会收尾 | 调大；多轮时要按整 episode 预算 |
| 4 | `temperature` 只调了生成没调损失 | 前后不一致，分布对不上 | TRL 里同一个 temperature 两处都用；自己实现时注意 |
| 5 | 数据里塞了 `completion` | 被当成普通列忽略，或者报字段错 | 蒸馏数据集**只放 prompt** |
| 6 | 教师太大、学生太小 | loss 降不下去（容量差太多） | 教师大 1 个量级即可 |
| 7 | 以为 loss 到 0 才算成功 | 永远等不到 | 学生追不上教师是常态，看结构与稳定性 |

---

## 10. 练习与自测

1. **看分布**：跑完整流程，找到 §5.5 那张表里 `p_教师 - p_学生` 最大的 3 个 token。它们是什么？通常是结构符号（`<tool_call>`、`{`、换行）——说明学生在"格式"上离教师最远。
2. **换 β**：`--beta 0.0` 与 `--beta 1.0` 各训一次（40 步即可），同一个 prompt 采 5 次，比较答案的**多样性**。前向 KL 应该更散。
3. **换温度**：`--temperature 2.0`，观察 `entropy` 与生成质量。温度太高教师分布变平，学生学到的是"含糊"。
4. **数据不加答案**：故意往 JSONL 里加一个 `completion` 字段，看脚本/TRL 怎么处理——理解"蒸馏不需要答案"这件事。

---

## 11. 一句话带走

> 蒸馏 = **"把教师脑子里的概率表复印给学生"**。它比 SFT 信息密度高，但天花板就是教师；
> 而且学生只在自己写出来的路径上受教——这正是它既高效又脆弱的原因。

---

[← 上一节：监督微调 SFT](../01-监督微调-SFT/README.md) · [总目录](../README.md) · [下一节：强化学习 GRPO →](../03-强化学习-GRPO/README.md)
