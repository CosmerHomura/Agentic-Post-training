# 01 · 监督微调（SFT）

[← 上一节：开始之前](../00-开始之前/README.md) · [总目录](../README.md) · [下一节：蒸馏 →](../02-蒸馏/README.md)

---

## 1. 一句话定义

**把"一个更强的模型是怎样一步步解决任务"的完整录像，当成标准答案让它背下来。**

这是整条后训练流水线的**第 1 步**，也是后面所有步骤的地基：没有 SFT，模型连"工具调用"的格式都不会写，RL 阶段根本无从采样。

---

## 2. 原文译文 · 1. Supervised fine-tuning (SFT)

> 模型学会复制 **agent 轨迹**——也就是 agent 解决一个任务的过程记录，其中包含每一次工具调用及其结果，通常由更强的模型产生。这一步便宜且稳定。
>
> 模型只会见到**正确的**轨迹，永远见不到它自己的尝试。当它在解题过程中犯了错，它并没有学过如何从错误中恢复。**On-policy 蒸馏**和 **RL** 通过在模型自己的输出上训练来解决这个问题。

### 原文推荐资源

| 类型 | 链接 |
|---|---|
| Watch | [Training Agents 1: fine-tune a coding agent](https://www.youtube.com/watch?v=rNgUoH7Wbv8)（[代码](https://huggingface.co/burtenshaw/gemma4-pi-mono-youtube-livestream-1-scripts)） |
| Read | [Training Agents: Class 1, SFT from scratch](https://x.com/ben_burtenshaw/status/2067615361428545566) |
| Docs | [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) |

---

## 3. 训练信号从哪来

| 环节 | 内容 |
|---|---|
| **输入** | `(system prompt + 工具列表 + 用户任务 + 完整正确轨迹)` |
| **模型产出** | 每个位置的下一个 token 分布 |
| **监督信号** | 数据里那一个**确定的 token**（硬标签，one-hot） |
| **损失** | 只在 assistant 的 token 上算的交叉熵：`-Σ log p(y_t)` |
| **产物** | 一个会模仿该轨迹（格式、顺序、风格）的模型 |

```
数据：   "先 read_file → 看到 bug → write_file → run_tests → 通过"
                    │
                    ▼
训练：   强制模型在每个位置都预测出数据里的那个 token
                    │
                    ▼
效果：   模型学会「这个任务该按这个顺序、用这些工具、这样写参数」
         但没学会「如果我刚才猜错了怎么办」（因为数据里从没有错误）
```

---

## 4. 输入格式：一条 agent 轨迹长什么样

### 4.1 SFTTrainer 支持的 4 种数据集格式

| 格式 | 列 | 适用 |
|---|---|---|
| 标准 LM | `text` | 纯文本续写 |
| 对话 LM | `messages` | **agent 轨迹就用这个**（角色 + 工具调用都在里面） |
| 标准 prompt-completion | `prompt`, `completion` | 单轮问答 |
| 对话 prompt-completion | `prompt`(messages), `completion`(messages) | 单轮、要求只对 completion 算 loss |

> agent 轨迹必须用 **对话格式 + `tools` 列**，因为一条轨迹里有多轮 assistant/tool 交替，单一 `completion` 表达不了。
> 官方文档也明确：对话格式下 TRL 会自动套用 tokenizer 的 chat template。

### 4.2 逐字段说明（对话格式）

| 字段 | 类型 | 含义 | 注意 |
|---|---|---|---|
| `messages` | `list[dict]` | 完整对话 | 每项 `{"role": ..., "content": ...}` |
| `messages[].role` | `system` / `user` / `assistant` / `tool` | 谁在说话 | `tool` 消息必须带 `name` |
| `messages[].content` | `str` | 文本 | assistant 只调工具时可以写 `""` |
| `messages[].tool_calls` | `list[dict]` | assistant 发起的工具调用 | `{"type":"function","function":{"name","arguments"}}` |
| `messages[].name` | `str` | 工具名 | 仅 `role="tool"` 时用 |
| `tools` | `list[dict]` | **工具 schema 列表** | 会被渲染进 prompt，告诉模型有哪些工具 |

### 4.3 真实示例：`data/sft_agent_trace.jsonl`（第 1 行，节选）

```json
{
  "tools": [
    {"type": "function", "function": {
      "name": "read_file",
      "description": "读取仓库中某个文件的完整内容。",
      "parameters": {"type": "object",
                     "properties": {"path": {"type": "string"}},
                     "required": ["path"]}}}
  ],
  "messages": [
    {"role": "system", "content": "你是一个编程 agent。你可以读文件、写文件、跑测试……"},
    {"role": "user",   "content": "calculator.py 里的测试失败了，请修好它。"},
    {"role": "assistant", "content": "先看代码和测试。",
     "tool_calls": [{"type": "function", "function": {
        "name": "read_file", "arguments": {"path": "calculator.py"}}}]},
    {"role": "tool", "name": "read_file",
     "content": "def divide(a, b):\n    return a / b\n"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "read_file",
                                          "arguments": {"path": "test_calculator.py"}}}]},
    {"role": "tool", "name": "read_file", "content": "from calculator import divide\n..."},
    {"role": "assistant", "content": "除零没有抛 ValueError，改掉。", "tool_calls": [
        {"type": "function", "function": {"name": "write_file",
          "arguments": {"path": "calculator.py", "content": "def divide(a, b):\n    ..."}}}]},
    {"role": "tool", "name": "write_file", "content": "written 96 bytes to calculator.py"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "run_tests", "arguments": {}}}]},
    {"role": "tool", "name": "run_tests", "content": "2 passed in 0.31s"},
    {"role": "assistant", "content": "在 divide 中增加了除零检查，两个测试都通过了。"}
  ]
}
```

> 一行 JSONL = 一条完整 episode。注意 assistant 有两类消息：
> **带 `tool_calls` 的**（决定下一步动作）和**纯文本的**（总结、推理），两种都要出现在数据里。

### 4.4 用代码生成 `tools` 列（避免手写 schema 出错）

```python
import json
from transformers.utils import get_json_schema

def read_file(path: str) -> str:
    """读取仓库中某个文件的完整内容。

    Args:
        path: 相对仓库根目录的文件路径。
    """
    ...

tools = [{"type": "function", "function": get_json_schema(read_file)}]
```

---

## 5. 输入输出：从 messages 到 labels

这是 SFT 最容易搞错的地方，也是"为什么我的模型只会复述用户问题"的根源。

### 5.1 第 1 步：模板把 messages 渲染成一条 token 序列

```
<|im_start|>system
你是一个编程 agent……<|im_end|><|im_start|>user
calculator.py 里的测试失败了……<|im_end|><|im_start|>assistant
先看代码和测试。
<tool_call>{"name":"read_file","arguments":{"path":"calculator.py"}}</tool_call><|im_end|>
<|im_start|>tool
def divide(a, b):
    return a / b
<|im_end|><|im_start|>assistant
……
```

### 5.2 第 2 步：生成 labels（关键！）

```
               system + user + tools          assistant         tool 结果        assistant
            ┌──────────────────────┐  ┌──────────────────┐  ┌──────────┐  ┌──────────────┐
input_ids = │ 151644 8948 ... 151645│ │ 872 198 ...151645│ │ ...      │  │ ...          │
            └──────────────────────┘  └──────────────────┘  └──────────┘  └──────────────┘
labels    = │ -100  -100 ...  -100  │ │ 872 198 ...151645│ │ -100 ... │  │ ...          │
            └──────────────────────┘  └──────────────────┘  └──────────┘  └──────────────┘
                  ↑ 不算 loss              ↑ 算 loss          ↑ 不算 loss      ↑ 算 loss
```

三条规则：

1. **prompt（system / user / 工具 schema）→ `-100`**，不算 loss。否则模型会学着"生成用户的提问"。
2. **assistant 的 token（含 tool_call 的结构与参数）→ 保留原 id**，算 loss。
3. **`role="tool"` 的结果 → `-100`**，不算 loss。工具输出不是模型该生成的东西。

> 这三条正是 `assistant_only_loss=True` 做的事：它要求 chat template 用
> `{% generation %}...{% endgeneration %}` 把 assistant 段标出来，只有这段参与 loss。
> Qwen3 等已知模型家族，TRL 会自动替换成带标记的模板。

### 5.3 第 3 步：实际喂进模型的张量

| 张量 | 形状 | 内容 |
|---|---|---|
| `input_ids` | `[1, L]` | 上图的 token id（packing 时多段拼在一起） |
| `attention_mask` | `[1, L]` | 真实 token 为 1，padding 为 0 |
| `labels` | `[1, L]` | 上图的 label，`-100` 表示忽略 |

模型输出 `logits: [1, L, V]`，框架内部做 `shift_logits` / `shift_labels`（右移一位预测下一个 token），再对非 `-100` 的位置求平均交叉熵。

### 5.4 一句话总结这张图

> **"你在数据里写了什么，模型就学什么；你没写的（-100），它一律不学。"**
> 想让模型学会"调用工具后等待结果再继续"，就必须让 `tool_call` 和它之后的 assistant 续写都在 generation 段内。

---

## 6. 代码与参数

完整脚本：[`code/train_sft.py`](code/train_sft.py)

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("json", data_files="../data/sft_agent_trace.jsonl", split="train")

args = SFTConfig(
    output_dir="./ckpt/sft-agent",
    assistant_only_loss=True,   # ★ 只对 assistant 算 loss
    packing=True,               # 短样本拼包，提升吞吐
    max_length=4096,            # agent 轨迹长，别设太小
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
)

trainer = SFTTrainer(model="Qwen/Qwen3-0.6B", args=args, train_dataset=dataset)
trainer.train()
trainer.save_model("./ckpt/sft-agent")
```

### 关键参数速查

| 参数 | 作用 | 建议 |
|---|---|---|
| `assistant_only_loss` | 只在 assistant token 上算 loss | agent 轨迹**必开** |
| `completion_only_loss` | prompt-completion 格式下只算 completion | 单轮问答用 |
| `packing` | 多条样本拼进一个序列 | 短样本多时开；`max_length` 要够 |
| `max_length` | 单条序列上限 | ≥ 最长轨迹；截断会把 assistant 段截没 |
| `loss_type="dft"` | 只用"模型本来就预测对的 token"算 loss | 怕过拟合时试 |
| `chat_template_path` | 指定带 `{% generation %}` 的自定义模板 | 模板缺标记时用 |
| `eos_token` | 终止符 | Qwen2.5 系需设 `"<|im_end|>"` |
| `learning_rate` | 学习率 | 全参 1e-5~2e-5；LoRA 1e-4 量级 |

### 如果要自己改 chat template

在 Jinja 模板里把 assistant 那段包起来即可：

```jinja
{% if message['role'] == 'assistant' %}
    {% generation %}{{ message['content'] }}{% endgeneration %}
{% endif %}
```

（真实模板还要处理 `tool_calls` 的渲染与 `tools` 列表，建议直接改 Qwen3 的模板。）

---

## 7. 产物与验收

### 7.1 训练日志里看什么

| 指标 | 含义 | 健康表现 |
|---|---|---|
| `loss` | 只覆盖 assistant token 的交叉熵 | 稳定下降；0.3~1.0 区间常见 |
| `mean_token_accuracy` | 模型 top-1 猜中的比例 | 随 loss 上升（60%→85%+） |
| `entropy` | 输出分布熵 | 缓降；掉到 0 附近=过拟合/塌缩 |
| `num_tokens` | 累计训练 token 数 | 用来估"每个 epoch 实际学了多少" |
| `grad_norm` | 梯度范数 | 别持续爆炸（>10） |
| `learning_rate` | 当前 lr | 确认 warmup/cosine 生效 |

### 7.2 离线自测（比 loss 更可信）

训完后让模型在**没见过的任务**上自由跑一次，看三件事：

```
□ 格式：会不会正确输出 tool_call 结构（而不是把 JSON 写进正文）
□ 顺序：会不会先读再改（而不是直接 write_file 瞎猜内容）
□ 停止：会不会在该结束时输出终止符，而不是无限调用工具
```

### 7.3 产物

- `./ckpt/sft-agent/`：可继续做蒸馏 / RL 的**起点模型**
- 一张"轨迹数据集"：后面第 4 步做环境时，可以拿它当行为参考

---

## 8. 常见坑

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | 没传 `tools` 列 | 模型在正文里写 JSON，而不是结构化 tool_call | 把工具 schema 放进 `tools` 列 |
| 2 | `arguments` 类型不对 | 模板渲染成 Python `repr`（单引号），解析失败 | 统一成 dict 或 JSON 字符串，与模板要求一致 |
| 3 | 没开 `assistant_only_loss` | 训完模型爱复述用户问题 | 开启，并确认模板有 `{% generation %}` |
| 4 | 把工具结果也算了 loss | 模型开始"编造工具输出" | tool 消息必须落在 generation 段外 |
| 5 | `max_length` 太小 | loss 变成 0 / NaN（整条 assistant 被截掉） | 调大，或过滤超长样本 |
| 6 | 数据只有成功轨迹 | 一遇错就崩（原文明确指出的 SFT 局限） | 这是 SFT 的天花板，靠第 2/3 步解决 |
| 7 | 只训 1 个 epoch 就评估 | 格式还没学会 | 至少 2–3 epoch，看 `mean_token_accuracy` |
| 8 | 学不会停止 | 生成不收敛到 EOS | 确认 EOS 在 generation 段内，且训练数据都以 EOS 结尾 |

---

## 9. 练习与自测

1. **改数据**：给 `sft_agent_trace.jsonl` 加第 3 条轨迹，要求它**先跑测试看到报错、再读文件**（顺序与现有样例相反）。观察模型是否能学到"先看错误信息"这个策略。
2. **验证 mask**：用 tokenizer 把第 1 条样本渲染出来，打印 `input_ids` 与 `labels`，标出哪几段是 `-100`。确认 `role="tool"` 那段确实是 `-100`。
3. **对照实验**：同一份数据跑两次，一次 `assistant_only_loss=True`，一次 `False`。用同一个 prompt 推理，比较输出差异——你应该能直观看到"复述用户问题"的现象。

---

## 10. 一句话带走

> SFT 教的是**"照着做"**：格式、顺序、语气都能学，但它永远看不到自己犯错，所以学不会**"错了怎么办"**。
> 这正是下一章（蒸馏）和第三章（RL）要补的洞。

---

[← 上一节：开始之前](../00-开始之前/README.md) · [总目录](../README.md) · [下一节：蒸馏 →](../02-蒸馏/README.md)
