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

## 4. 数据格式：一条 agent 轨迹长什么样

### 4.1 SFTTrainer 支持的 4 种数据集格式

| 格式 | 列 | 适用 |
|---|---|---|
| 标准 LM | `text` | 纯文本续写 |
| 对话 LM | `messages` | **agent 轨迹就用这个**（角色 + 工具调用都在里面） |
| 标准 prompt-completion | `prompt`, `completion` | 单轮问答 |
| 对话 prompt-completion | `prompt`(messages), `completion`(messages) | 单轮、要求只对 completion 算 loss |

> agent 轨迹必须用 **对话格式 + `tools` 列**，因为一条轨迹里有多轮 assistant/tool 交替，单一 `completion` 表达不了。

### 4.2 逐字段说明（对话格式）

| 字段 | 类型 | 含义 | 注意 |
|---|---|---|---|
| `messages` | `list[dict]` | 完整对话 | 每项 `{"role": ..., "content": ...}` |
| `messages[].role` | `system` / `user` / `assistant` / `tool` | 谁在说话 | `tool` 消息带上 `name` 便于自己溯源（Qwen3 模板不读它） |
| `messages[].content` | `str` | 文本 | assistant 只调工具时可以写 `""` |
| `messages[].tool_calls` | `list[dict]` | assistant 发起的工具调用 | `{"type":"function","function":{"name","arguments"}}` |
| `messages[].name` | `str` | 工具名 | 仅 `role="tool"` 时用 |
| `tools` | `list[dict]` | **工具 schema 列表** | 会被渲染进 **system** 消息，告诉模型有哪些工具 |

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

> 一行 JSONL = 一条完整 episode。assistant 有两类消息：**带 `tool_calls` 的**（决定下一步动作）和**纯文本的**（总结、推理），两种都要出现在数据里。

### 4.4 用代码生成 `tools` 列（避免手写 schema 出错）

```python
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

## 5. 全链路实拍：一条轨迹进模型前，变四次形

```
JSONL 记录 ──chat template──► 一段纯文本 ──tokenizer──► token id 序列
    ──打掩码──► labels（学的保留 id，不学的 -100）──collate──► batch 张量 ──► loss
```

下面就按这个顺序，把 `python train_sft.py --inspect` 的真实输出贴出来（模型 `Qwen/Qwen3-0.6B`，数据 `data/sft_agent_trace.jsonl` 第 1 条）。**这些数字都可以在你自己机器上复现。**

### 5.1 阶段 1：读 JSONL

```
  文件：.../01-监督微调-SFT/data/sft_agent_trace.jsonl
  行数：2

   #  role         content 长度  tool_calls
  --------------------------------------------------------------
   0  system               46  -
   1  user                 52  -
   2  assistant             8  read_file
   3  tool                 35  -
   4  assistant             0  read_file
   5  tool                223  -
   6  assistant            20  write_file
   7  tool                 33  -
   8  assistant             0  run_tests
   9  tool                 17  -
  10  assistant            49  -

  第 1 条记录的 tools 列（渲染进 system 消息里的函数签名）：
    - read_file(path)     # 读取仓库中某个文件的完整内容。
    - write_file(path, content)  # 把内容写入仓库中的文件，覆盖原内容。
    - run_tests()         # 运行仓库的 pytest 测试套件。
```

看出结构了：**user 只出现 1 次，assistant 出现 5 次**——中间 4 次是"调工具"，最后 1 次是"给结论"。

### 5.2 阶段 2：套 chat template（JSON → 纯文本）

`tokenizer.apply_chat_template(messages, tools=...)` 渲染出 **2447 个字符**。原样打印（`▸` 是脚本加的标记，表示模板插入的角色分隔符）：

```
▸ <|im_start|>system
  你是一个编程 agent。你可以读文件、写文件、跑测试。修好失败的测试后，用一句话总结改动。

  # Tools

  You may call one or more functions to assist with the user query.

  You are provided with function signatures within <tools></tools> XML tags:
  <tools>
  {"type": "function", "function": {"name": "read_file", "description": "读取仓库中某个文件的完整内容。", "parameters": {...}}}
  {"type": "function", "function": {"name": "write_file", "description": "把内容写入仓库中的文件，覆盖原内容。", "parameters": {...}}}
  {"type": "function", "function": {"name": "run_tests", "description": "运行仓库的 pytest 测试套件。", "parameters": {"type": "object", "properties": {}}}}
  </tools>

  For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
  <tool_call>
  {"name": <function-name>, "arguments": <args-json-object>}
  </tool_call><|im_end|>
▸ <|im_start|>user
  calculator.py 里的测试失败了，请修好它。测试文件是 test_calculator.py。<|im_end|>
▸ <|im_start|>assistant
  先看代码和测试。
  <tool_call>
  {"name": "read_file", "arguments": {"path": "calculator.py"}}
  </tool_call><|im_end|>
▸ <|im_start|>user            ← 注意！工具结果在 Qwen3 模板里是 user 轮
  <tool_response>
  def divide(a, b):
      return a / b

  </tool_response><|im_end|>
▸ <|im_start|>assistant
  <tool_call>
  {"name": "read_file", "arguments": {"path": "test_calculator.py"}}
  </tool_call><|im_end|>
  ...
▸ <|im_start|>assistant
  <think>

  </think>

  在 divide 中增加了除零检查，b == 0 时抛出 ValueError，两个测试都通过了。<|im_end|>
```

这里有 **四个必须亲眼看到**的事实，每一个都会影响训练结果：

| # | 事实 | 后果 |
|---|---|---|
| 1 | `tools` 不是单独一段，是被塞进 **system 消息**（`# Tools` + `<tools>` JSON + 一段调用格式说明） | 换模型就得换写法；工具描述写得含糊，模型就选错工具 |
| 2 | `role="tool"` 被渲染成 **`<|im_start|>user` + `<tool_response>…</tool_response>`**，不是 `<|im_start|>tool` | 想靠"只学 user 轮之后"来打掩码会错；工具结果天然属于"上下文" |
| 3 | assistant 的 `tool_calls` 被渲染成 `<tool_call>{"name":…,"arguments":…}</tool_call>`，**参数是 JSON** | 训练数据里 `arguments` 必须是 dict（或合法 JSON 串），写成 Python `repr` 就学成单引号 |
| 4 | 最后一条 assistant 前面被插了 **空的 `<think>\n\n</think>`** | Qwen3 的"非思考模式"就是用空 think 块表示；训练数据保持一致即可，别指望它凭空学会思考 |

### 5.3 阶段 3：tokenize → 打掩码 → labels

把上面的文本切成 token，共 **650 个**。然后回答一个问题：**这 650 个 token，哪些该学？**

```
  已给模板打 generation 补丁
  总 token：650    需要学的 token：166（25.5%）

  掩码条（█ = 要学，· = 不学，每行 100 个 token）：
       0 ····································································································
     100 ····································································································
     200 ····································································································
     300 ··············································████████████████████████████··················
     400 ·····████████████████████████··································································█████
     500 ██████████████████████████████████████████████████████████████····················█████████████████·
     600 ····················██████████████████████████████
```

**为什么前面 300 多个 token 全是 `·`？** 因为那是 system + 工具 schema + 用户提问——全是"给定上下文"，让模型去预测它们只会教它复述用户。

掩码翻转的精确位置（表里第 1 列是下标，第 3 列是 label）：

```
   idx  token_id     label  学   token
  ----------------------------------------------------------
   348      1773      -100  ·   。
   349    151645      -100  ·   <|im_end|>
   350       198      -100  ·   \n
   351    151644      -100  ·   <|im_start|>     ← assistant 的角色头，不算
   352     77091      -100  ·   assistant       ← 同上
   353       198      -100  ·   \n              ← 同上
   354     60726     60726  █   先              ← 从这里开始学
   355     50930     50930  █   看
   356     46100     46100  █   代码
   357     33108     33108  █   和
   358     81705     81705  █   测试
   359      8997      8997  █   。\n
   360    151657    151657  █   <tool_call>     ← 工具调用的结构也要学
   361       198       198  █   \n
   362      4913      4913  █   {"
   363       606       606  █   name
   ...
   647      1773      1773  █   。
   648    151645    151645  █   <|im_end|>     ← 学会停下来
   649       198       198  █   \n
```

三条规则：

1. **system / user / 工具 schema / `tool_response` → `-100`**，不算 loss。
2. **assistant 轮的全部 token → 保留原 id**，包括 `<tool_call>` 结构、参数 JSON、结尾的 `<|im_end|>`。
3. **padding 也必须是 `-100`**，否则模型会去学"怎么补空格"。

> 「哪些 token 是 assistant 的」这件事，transformers 靠模板里的 `{% generation %}…{% endgeneration %}` 标记来判断。
> Qwen3 的模板**没有**这对标记——直接调 `return_assistant_tokens_mask=True` 会得到**全 0 的掩码**（外加一条 warning）。
> TRL 的 `assistant_only_loss=True` 内部做的就是"给已知模型家族的模板打补丁"，脚本里 `patch_template()` 是同一个动作的最小实现。

### 5.4 阶段 4：拼成 batch（模型真正看到的东西）

两条样本 pad 到同一长度后：

```
  样本数：2   序列长度（pad 后）：650

  input_ids         [2, 650]  int64   # pad=151643
  attention_mask    [2, 650]  int64   # pad 处为 0
  labels            [2, 650]  int64   # 不学的位置 = -100

  第 1 条样本的头 12 个位置：
   idx   input_ids      labels  attn
     0      151644        -100     1
     1        8948        -100     1      ← system
     2         198        -100     1
     3       56568        -100     1      你
     4      101909        -100     1      是一个
     ...
```

模型输出 `logits: [2, 650, 151936]`，框架内部把 logits 右移一位、按 `labels != -100` 求平均交叉熵。**也就是说：loss 只由那 166 个位置决定，其余 484 个位置对梯度没有贡献。**

### 5.5 一句话总结这四次变形

> **"你在数据里写了什么，模型就学什么；你没写的（-100），它一律不学。"**
> 想让模型学会"调用工具后等结果再继续"，就必须让 `tool_call` 和它之后的 assistant 续写都落在监督段里。

---

## 6. 损失怎么算（手算一遍）

第 5 节讲的是「数据变成了什么」，这一节讲「这些数字最后怎么变成一个 loss」。
`python train_sft.py --explain-loss` 会把下面的表原样打印出来（不加载模型，1 秒跑完）。

### 6.1 一个位置：交叉熵就是「正确 token 概率的负对数」

模型在每个位置吐出 151936 个分数（logits），softmax 之后变成一个概率分布。
我们只关心**正确那个 token** 拿到了多少概率 `p`：

```
  loss_这一个位置 = -ln(p)
```

| 正确 token 的概率 `p` | `-ln(p)` | 直觉 |
|---|---|---|
| 0.99 | 0.010 | 几乎确定 → 几乎不罚 |
| 0.90 | 0.105 | 很确定 |
| 0.50 | 0.693 | 在犹豫 |
| 0.20 | 1.609 | 猜错了 |
| 0.01 | 4.605 | 完全跑偏 → 罚得很重 |

这张表解释了 loss 的量级：SFT 正常的 loss 从 **2 左右降到 0.3 左右**，就是「正确 token 的平均概率从约 0.13 涨到约 0.74」。

### 6.2 一条轨迹：166 个位置各算一次，再平均

假设某条轨迹里 5 个被监督的 token，模型给它们的概率分别是：

```
  token 1: p = 0.98   →   -ln(p) = 0.020
  token 2: p = 0.95   →   -ln(p) = 0.051
  token 3: p = 0.90   →   -ln(p) = 0.105
  token 4: p = 0.80   →   -ln(p) = 0.223
  token 5: p = 0.50   →   -ln(p) = 0.693
  ─────────────────────────────────────────
  平均 = (0.020+0.051+0.105+0.223+0.693)/5 = 0.218
```

**只有这 5 个位置参与**。真实那条轨迹是 166 个位置，公式一样：

```
  loss = (1 / 166) · Σ_{t ∈ 监督段} -ln p(token_t)
```

分母是 **166 而不是 650** —— 这就是 5.3 节那张掩码条的真正作用。分母搞错（比如忘了 `assistant_only_loss`）会让 loss 数值凭空小一半，看着像「学得特别好」。

### 6.3 「右移一位」到底是什么意思

语言模型在位置 `t` 输出的是**下一个** token 的分布，所以算 loss 时要把 label 往左错一位：

```
  位置 0   输入 <|im_start|>       → 预测下一个：assistant
  位置 1   输入 assistant          → 预测下一个：换行
  位置 2   输入 换行                → 预测下一个：先
  位置 3   输入 先                  → 预测下一个：调
  位置 4   输入 调                  → 预测下一个：用
```

所以 `labels[t]` 就是 `input_ids[t+1]`，整条 label 序列等于「输入左移一位、末尾补 -100」：

```
  input_ids :  <|im_start|>  assistant  换行  先  调  用  </tool_call>  ...
  labels    :  assistant     换行       先    调  用  </tool_call>  ...  -100
  loss_mask :  0             0          0     1   1   1          1       0
```

只有 `loss_mask = 1` 的位置进入损失；`0` 的位置（system、user、工具返回、padding）全部跳过。

被 mask 的位置（prompt、tool 结果、padding）把 label 写成 `-100`，框架直接跳过它们 —— **不参与求和，也不参与平均**。

### 6.4 整个 batch

一个 batch 里 2 条样本、每条 166 个监督位置：

```
  loss = ( Σ_{i=1..2} Σ_{t ∈ 监督段} -ln p_{i,t} ) / (2 × 166)
```

梯度是最朴素的那种：**每个被监督的 token 各自把「自己这个 token 的概率」往上推一点**，互不干扰。所以：

- 想让它学会某种 token 序列 → 那段必须是 `label != -100`；
- 不想让它学（比如工具返回的 JSON）→ 那段必须 `-100`；
- 数量少的片段（比如 3 个 token 的 `<|im_end|>`）在 166 个位置里权重很小，想让模型「学会停」，得靠足够多的样本重复。

### 6.5 一句话记住

> **loss = 「被监督的每个 token，模型给它的概率有多低」的平均。**
> 掩码决定**分子和分母里算哪些位置**，右移决定**每个位置在预测谁**。

---

## 7. 跑起来

脚本：[`code/train_sft.py`](code/train_sft.py)（约 660 行，四个阶段 + 训练前/后评测全部可打印）

### 7.1 两套数据，别混用

| | 本地解剖样本 | 官方训练集 |
|---|---|---|
| 位置 | `data/sft_agent_trace.jsonl`（2 条） | `--dataset trl-lib/Capybara`（1.6 万条对话） |
| 用来看 | 模板长什么样、掩码打在哪、token 怎么变 label | **训练效果** |
| 能不能训出能力 | 不能。2 条样本训 20 步只会把这 2 条背下来 | 能，配 held-out 评测就能看到 -ln p 下降 |

第 5 节那些数字（650 token / 166 监督 / 24.4%）都来自解剖样本，它的作用是让你**看清数据处理**，不是让你相信训练有效。
要看效果，就得用官方数据集或者你自己采集的上千条轨迹。

### 7.2 命令

```bash
cd 01-监督微调-SFT/code

# (1) 只跑数据链路：不需要 GPU、不需要 trl（只要 transformers），先确认数据没写错
python train_sft.py --inspect

# (1).5 不加载模型，把第 6 节的交叉熵手算打一遍
python train_sft.py --explain-loss

# (2) 用本地 2 条解剖样本跑通全流程（验证管线，不代表效果）
python train_sft.py --epochs 1 --max-steps 20

# (3) 真正训练：官方数据集 2000 条，留 100 条做 held-out 评测
python train_sft.py --dataset trl-lib/Capybara --limit 2000 --eval-size 100

# 单卡 24G 的稳妥配置
python train_sft.py --dataset trl-lib/Capybara --limit 5000 --eval-size 100 \
    --model Qwen/Qwen3-0.6B --epochs 1 --gen-tokens 160
```

`--limit` 取多少？先 2000 条跑通（约 20–40 分钟），确认 -ln p 在降，再加到 5000–20000。

依赖（一次装完）：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "trl>=1.16.0" transformers datasets accelerate
```

| 产物 | 内容 |
|---|---|
| `runs/sft/report.md` | 总报告：环境、held-out 前/后 -ln p、loss 表、ASCII 损失图、生成对比 |
| `runs/sft/tokens.txt` | 全量 token / label / mask 表（解剖样本，650 行） |
| `runs/sft/loss.csv` | 每步 loss，可拿去画图 |
| `ckpt/sft-agent/` | 训练好的权重，后面几步的起点 |

### 7.3 怎么判断「真的有效果」

别只看 loss 曲线，看这三个数（report.md 里都有）：

| 指标 | 从哪来 | 健康 |
|---|---|---|
| **held-out -ln p** | 100 条没参与训练的样本，训练前后各算一次 | 下降 15% 以上（如 2.1 → 1.4） |
| **token_acc** | 训练日志 | 0.4 → 0.8 以上 |
| **生成对比** | 同样的 prompt，前后各生成一次 | 训练后能套上 `<tool_call>{…}</tool_call>` 格式 |

前两个是数，第三个是眼睛；三个都变好才叫学会了。

### 7.4 训练时终端长这样

`--inspect` 之后进入训练，每个 logging 步打印一行（`logging_steps=1`）：

```
  模型：Qwen/Qwen3-0.6B   参数量：596M
  device：cuda:0   bf16：True

  step   1  loss 2.1743  lr 3.3e-07  grad 4.512  token_acc 0.412
  step   2  loss 1.9806  lr 6.7e-07  grad 3.221  token_acc 0.488
  step   3  loss 1.8122  lr 1.0e-06  grad 2.884  token_acc 0.531
  ...
```

<!-- 待回填：把 report.md 里真实的 loss 表 / 损失图贴到这里 -->

训练的每一行都要看三件事：

| 信号 | 健康 | 不健康 |
|---|---|---|
| `loss` | 单调下降后趋稳 | 抖动剧烈 → lr 太大；不降 → 掩码错了（全是 -100） |
| `token_acc` | 稳步升（0.4 → 0.8+） | 停在 0.3 附近 → 数据太少/太难 |
| `grad` | 平稳（0.1–5） | 尖峰 >20 → 梯度爆炸，开 `max_grad_norm` 或降 lr |

### 7.5 损失图（终端里也能看）

`report.md` 里会附一张 ASCII 走势图，不装 matplotlib 也能判断"到底在不在学"：

```
  loss 走势 (min 0.412 / max 2.174)
   2.174 |●
   1.881 | ●
   1.587 |  ●
   1.294 |   ●●
   1.000 |     ●●
   0.706 |       ●●
   0.412 |         ●●●
         +----------------------------------------------
          step 1 → 36
```

<!-- 待回填：真实损失图 -->

### 7.6 训练前后对比

同一批 prompt，训练前（基座）和训练后（SFT 完）各生成一次：

```
  ── 同一类任务（修 bug） ──
  prompt（最后一句）：calculator.py 里的测试失败了，请修好它。测试文件是 test_calculator.py。
  模型输出：
    ...
```

<!-- 待回填：report.md 里 before/after 两段原文 -->

怎么读这段对比：

| 看什么 | 训练前（基座） | 训练后（SFT） |
|---|---|---|
| 有没有 `<tool_call>` | 一般没有，或写成正文 Markdown | 出现结构化 `<tool_call>` |
| 参数是不是 JSON | 常是自然语言 | `{"path": "calculator.py"}` |
| 会不会等工具结果 | 不会（没有"轮"的概念） | 会先读再改，最后给结论 |
| 会不会停 | 继续瞎编 | 输出 `<|im_end|>` 收尾 |

> 注意：**2 条数据只能验证流水线通不通，不能真的教会能力。** 想看到明显变化，至少 100+ 条同格式轨迹（见 00 章"数据从哪来"）。

---

## 8. 参数速查

| 参数 | 作用 | 建议 |
|---|---|---|
| `assistant_only_loss` | 只在 assistant token 上算 loss | agent 轨迹**必开** |
| `packing` | 多条样本拼进一个序列 | 短样本多时开；默认关，方便"一条轨迹=一条序列"对照 |
| `max_length` | 单条序列上限 | ≥ 最长轨迹；截断会把 assistant 段截没（本脚本 4096） |
| `loss_type="dft"` | 只用"模型本来就预测对的 token"算 loss | 怕过拟合时试 |
| `chat_template_path` | 指定带 `{% generation %}` 的自定义模板 | 模板缺标记、又不想让 TRL 自动改时用 |
| `eos_token` | 终止符 | Qwen2.5 系需设 `"<|im_end|>"` |
| `learning_rate` | 学习率 | 全参 1e-5~2e-5；LoRA 1e-4 量级 |

### 自己改 chat template 的话

在 Jinja 里把 assistant 那段包起来即可：

```jinja
{%- elif message.role == "assistant" %}{%- generation %}
    ...（角色头、内容、tool_calls、<|im_end|>）...
{%- endgeneration %}{%- elif message.role == "tool" %}
```

（真实模板还要处理 `tools` 列表与 `tool_calls` 渲染，建议就在 Qwen3 模板上改这两处，脚本里的 `patch_template()` 就是这么干的。）

---

## 9. 产物与验收

### 9.1 训练日志里看什么

| 指标 | 含义 | 健康表现 |
|---|---|---|
| `loss` | 只覆盖 assistant token 的交叉熵 | 稳定下降；0.3~1.0 区间常见 |
| `mean_token_accuracy` | 模型 top-1 猜中的比例 | 随 loss 上升（60%→85%+） |
| `entropy` | 输出分布熵 | 缓降；掉到 0 附近=过拟合/塌缩 |
| `num_tokens` | 累计训练 token 数 | 用来估"每个 epoch 实际学了多少" |
| `grad_norm` | 梯度范数 | 别持续爆炸（>10） |

### 9.2 离线自测（比 loss 更可信）

```
□ 格式：会不会正确输出 tool_call 结构（而不是把 JSON 写进正文）
□ 顺序：会不会先读再改（而不是直接 write_file 瞎猜内容）
□ 停止：会不会在该结束时输出终止符，而不是无限调用工具
```

### 9.3 产物

- `ckpt/sft-agent/`：可继续做蒸馏 / RL 的**起点模型**
- `runs/sft/report.md`：这次训练的证据（loss、前后对比），也是排查问题的第一手材料

---

## 10. 常见坑

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | 没传 `tools` 列 | 模型在正文里写 JSON，而不是结构化 tool_call | 把工具 schema 放进 `tools` 列 |
| 2 | `arguments` 类型不对 | 模板渲染成 Python `repr`（单引号），解析失败 | 统一成 dict 或 JSON 字符串 |
| 3 | 模板没有 `{% generation %}` | `assistant_masks` 全 0，或者 loss 直接 0 / NaN | 用 `patch_template()` 打补丁，或让 TRL 自动补（`assistant_only_loss=True`） |
| 4 | 把 `tool_response` 也算进 loss | 模型开始"编造工具输出" | 工具结果的 token 必须 `-100`（它是 user 轮） |
| 5 | padding 的 label 没设 `-100` | loss 虚高、学出一堆空格 | collator 里 pad 的 label 一律 `-100` |
| 6 | `max_length` 太小 | 整条 assistant 被截掉，loss 变 0/NaN | 调大，或过滤超长样本 |
| 7 | 忘了 Qwen3 的空 `<think>` 块 | 训完模型先吐一个空 think 再回答——其实正常 | 训练数据与推理保持一致即可 |
| 8 | 数据只有成功轨迹 | 一遇错就崩（原文明确指出的 SFT 局限） | 这是 SFT 的天花板，靠第 2/3 步解决 |
| 9 | 学不会停止 | 生成不收敛到 EOS | 确认 `<|im_end|>` 在监督段内，且每条轨迹都以它结尾 |

---

## 11. 练习与自测

1. **看掩码**：跑 `python train_sft.py --inspect`，打开 `runs/sft/tokens.txt`，找到第 4 条消息（`tool_response`）——确认它的每一行都是 `0`。再找到最后一条 assistant 的 `<think>`——确认它被监督了。
2. **改数据**：加第 3 条轨迹，让它**先跑测试看到报错、再读文件**（顺序与现有样例相反），重跑，看掩码条的形状有没有变、模型有没有学到这个顺序。
3. **对照实验**：把 `assistant_only_loss=True` 改成 `False` 再训一次，用同一个 prompt 推理——你应该能直观看到"复述用户问题"的现象。
4. **算比例**：把 `--epochs` 从 3 改成 1，比较 `num_tokens` 与 `166 × 2 × epoch`，确认你真的只学了该学的部分。

---

## 12. 一句话带走

> SFT 教的是**"照着做"**：格式、顺序、语气都能学，但它永远看不到自己犯错，所以学不会**"错了怎么办"**。
> 这正是下一章（蒸馏）和第三章（RL）要补的洞。

---

[← 上一节：开始之前](../00-开始之前/README.md) · [总目录](../README.md) · [下一节：蒸馏 →](../02-蒸馏/README.md)
