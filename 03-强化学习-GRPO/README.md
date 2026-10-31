[← 上一章：02 · 蒸馏](../02-蒸馏/README.md) ｜ [总目录](../README.md) ｜ [下一章：04 · RL 环境 →](../04-RL环境/README.md)

# 03 · 强化学习（GRPO）

> 原文小节：**3. Reinforcement learning (RL)** —— 用「一个分数」代替教师和答案轨迹。
> 这一章配了**可运行脚本**：`code/train_grpo.py`（全链路打印 + 损失手算 + 训练 + 前后对比）。

---

## 1. 一句话定义

给同一个问题采样 G 个答案，逐条打分，然后**让「比组内平均分高」的答案更可能出现、比平均分低的更不可能**。

不需要教师模型，也不需要标准答案轨迹 —— 只需要一个能判断对错的东西。这是从「抄」转向「练」的分水岭。

---

## 2. 原文译文 · 3. Reinforcement learning (RL)

> 在强化学习中，模型从对自己输出的一个分数里学习。它写一个答案，奖励函数给它打分，训练让高分答案变得更可能。这样就不再需要教师 —— 只要能校验答案就够了。
>
> 这一步用的算法是 **GRPO（Group Relative Policy Optimization）**。对每个 prompt，它采样若干个答案，给它们全部打分，然后提高那些得分高于组内平均的答案的概率。在组内比较答案，就告诉了模型哪些尝试更好，而不需要另外训练一个模型来估计这件事（这正是它相对 PPO 省掉 critic 的地方）。
>
> 这个信号比蒸馏弱：整条答案只有一分数，而不是每个 token 一个值，所以模型需要很多次尝试才能学到东西。用普通奖励函数时，每个答案只有一轮、只打一次分。而环境让模型可以行动多次，并看到每个动作的结果。

**原文推荐资源**

| 类型 | 链接 |
|---|---|
| Watch | [Training Agents 3: reinforcement learning](https://huggingface.co/collections/merve/training-agents-3-reinforcement-learning-6ad6aa71050342629dd0e3ec)（合集） |
| Read | [How frontier models train on outcomes in 2026](https://huggingface.co/blog/merve/agentic-rl-2026) |
| Project | [Fine-tuning a 350M model for structured outputs in 100 GRPO steps](https://huggingface.co/spaces/merve/grpo-with-trl-ifstruct) |
| Docs | [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) |

---

## 3. 训练信号从哪来

| 环节 | 内容 |
|---|---|
| 输入 | prompt + **判分依据**（标准答案、单测、正则…） |
| 模型产出 | G 条完整答案 + 每条的分数 |
| 监督信号 | **每条答案一个标量** `r_i`，组内归一化成优势 `A_i` |
| 损失 | 策略梯度：提高 `A>0` 的 token 概率、压低 `A<0` 的 |
| 和 SFT 的差别 | SFT 学「这个 token 一定对」；GRPO 只学「这条路整体更好」 |

**信号密度对比**（这就是原文说的「比蒸馏弱」）：

```
SFT        : 166 个 token，每个都有确定答案        → 强、稳，但只会模仿
蒸馏        : 每 token 一个教师分布（几千维）        → 最强，但要教师
GRPO       : 整条 1 个数（group 内相对比较）        → 弱、稀疏，但可以超越教师
```

---

## 4. 输入格式：只有 prompt，没有答案

`data/grpo_prompts.jsonl`：

```json
{"prompt": [{"role": "system", "content": "你是一个数学解题助手。先想，再答。"},
            {"role": "user", "content": "23 × 47 = ?\n请把最终答案写在 <answer></answer> 里。"}],
 "ground_truth": "1081"}
```

逐字段：

| 字段 | 类型 | 作用 | 注意 |
|---|---|---|---|
| `prompt` | `list[dict]` | 送进模型的对话（最后一条是 user） | 对话格式，训练时自动套 chat template |
| `ground_truth` | `str` | 判分用 | 名字随你定，会被当作 kwargs 传给奖励函数 |
| `tools`（可选） | `list[dict]` | 工具 schema | 需要多轮工具时才加 |
| `environment`（可选） | `str` | 多环境时的路由键 | 见第 4 章 |

**关键区别**：这里**没有 `completion` 字段**。答案由模型自己写，写 8 次就是 8 条不同的样本。

---

## 5. 奖励函数：唯一的「老师」

奖励函数就是一个普通 Python 函数：收一批 completion，返回**同样长度的 `list[float]`**。

```python
def accuracy_reward(completions, ground_truth=None, **kwargs):
    scores = []
    for comp, gt in zip(completions, ground_truth or [None] * len(completions)):
        content = comp[0]["content"] if isinstance(comp, (list, tuple)) else comp
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.S)
        got = m.group(1).strip() if m else ""
        scores.append(1.0 if got == str(gt).strip() else 0.0)
    return scores
```

**三条契约**（违反任何一条都会静默训坏）：

1. 返回 `list[float]`，长度 = completion 条数，**顺序一一对应** —— 返回单个 float 或标量会被当成「所有答案同分」。
2. 数据集里除 `prompt` 外的列会**按列名**传进函数（`ground_truth=...`）；签名里写 `**kwargs` 最安全。
3. 可以 sync，也可以 `async def`（多个异步奖励会并发跑）。

可用的关键字参数：`prompts`、`completions`、`completion_ids`、`trainer_state`、`log_extra(col, values)`、`log_metric(name, value)`、`environments`，以及数据集的其它列。

---

## 6. 全链路实拍

> 下面第 1～4 步是**结构示意**（真实跑一份，数字随采样变化）；
> 跑过 `python train_grpo.py` 后把 `runs/grpo/report.md` 发我，我回填成实测值。

### 6.1 第 1 步：读数据（没有答案）

```
文件：…/03-强化学习-GRPO/data/grpo_prompts.jsonl
行数：3

── 第 1 条 ──
  prompt      : 23 × 47 = ?  请把最终答案写在 <answer></answer> 里。…
  ground_truth: '1081'
  其他列      : 无

和 SFT 的区别：这里没有 assistant 轨迹，只有「什么算对」。
```

### 6.2 第 2 步：套模板 + tokenize（和 SFT 前半段完全一样）

```
<|im_start|>system
你是一个数学解题助手…<|im_end|>
<|im_start|>user
23 × 47 = ?…<|im_end|>
<|im_start|>assistant
```

到 `assistant\n` 为止都相同。**差别从下一步开始**：SFT 在这里塞人写好的轨迹，GRPO 让模型自己往下写 G 条。

### 6.3 第 3 步：采样 G 条并逐条打分

```
── 第 1 条 ──   <think>23×47 = 23×40 + 23×7 = 920 + 161</think>   <answer>1081</answer>
── 第 2 条 ──   <think>…</think> <answer>1071</answer>
── 第 3 条 ──   …<answer>1081</answer>
── 第 4 条 ──   23 × 47 = 1081                      ← 没按格式写
…

  i     format   accuracy   reward
  --------------------------------
  1        1.0        1.0      2.0
  2        1.0        0.0      1.0
  3        1.0        1.0      2.0
  4        0.0        0.0      0.0   ← 没格式 → 两个函数都判 0
```

注意第 4 条：**格式分不是装饰**。没有 `<answer>` 标签，正确性分也拿不到 —— 所以格式奖励会逼模型先学会「怎么说」，再学「说得对」。

### 6.4 第 4 步：组内归一化算优势

```
组内均值 mean = 1.25   标准差 std = 0.829

  i   reward    A_i   含义
  ------------------------------
  1      2.0   0.906   比平均好 → 提高这串 token 的概率
  2      1.0  -0.302   比平均差 → 压低
  3      2.0   0.906   比平均好 → 提高
  4      0.0  -1.509   最差 → 压得最狠
```

**为什么减均值、除标准差**：绝对分数受任务难度影响（全是 2 分的题和全是 0 分的题没法共享一套学习率）；组内相对化之后，每道题都变成「谁比谁好」，梯度尺度统一。

**退化组**：如果一组全对（`std = 0`），所有 `A_i = 0`，这一步**学不到任何东西**。所以选题要选「模型能做对一部分」的难度 —— 全对全错都是在浪费算力。

---

## 7. 损失怎么算（手算一遍）

这一步最容易含糊过去。下面全部用具体数字走一遍，`code/train_grpo.py --explain-loss` 会把这段原样打印出来。

### 7.1 四步总览

```
  (1) 采样 G 条        prompt ──► [c1, c2, …, cG]             （模型自己写）
  (2) 逐条打分         [c1..cG] ──► [r1, r2, …, rG]           （奖励函数）
  (3) 组内归一化       [r1..rG] ──► [A1, A2, …, AG]           A_i = (r_i-μ)/(σ+ε)
  (4) 逐 token 算损失   每个 completion 的每个 token 取一次 ratio，裁剪后乘 A_i，再平均
```

### 7.2 第 (3) 步的具体数字

取一组 4 条，奖励 `[1, 0, 1, 0]`：

```
  μ  = (1+0+1+0)/4 = 0.50
  σ  = sqrt(((1-.5)²+(0-.5)²+(1-.5)²+(0-.5)²)/4) = 0.50
  A  = [(1-.5)/.5, (0-.5)/.5, (1-.5)/.5, (0-.5)/.5]
     = [+1.0, -1.0, +1.0, -1.0]
```

对的两条各 `+1`，错的两条各 `-1` —— 正负对称，这就是 GRPO **不需要 critic** 的原因：基线直接由同组其它样本提供。

### 7.3 第 (4) 步：一个 token 的损失长什么样

`ratio = p_new(token) / p_old(token)`，即「更新后的模型对这个 token 的偏好」相对「采样时那个模型」涨了几倍。

对**正确**答案里某个 token（`A = +1`），假设 `p_old = 0.20`、`p_new = 0.26`：

```
  ratio        = 0.26 / 0.20 = 1.30
  未裁剪项      ratio · A           = 1.30 × 1   = 1.30
  裁剪项        clip(ratio,0.8,1.2) · A = 1.20 × 1 = 1.20
  损失贡献      -min(1.30, 1.20)              = -1.20
```

概率涨到 1.2 倍以后**不再奖励**（`min` 取小的那个），防止一步更新把策略推飞。这就是 PPO 的 clip。

对**错误**答案里同一个 token（`A = -1`）：

```
  ratio · A         = 1.30 × (-1) = -1.30
  clip 后           = 1.20 × (-1) = -1.20
  损失贡献          = -min(-1.30, -1.20) = +1.30
```

符号全反了：错误答案的概率一旦被推高，损失立刻变大，梯度把它压回去。

### 7.4 两条曲线（图）

把同一个 `ratio` 代进两种 `A`，算出来的损失是这样的（每行 0.05 一档，□ = 一格）：

```
  ratio       0.8   0.9   1.0   1.1   1.2   1.3   1.4   1.5
  A = +1     -0.80 -0.90 -1.00 -1.10 -1.20 -1.20 -1.20 -1.20   ← 涨到 1.2 就封顶
  A = -1     +0.80 +0.90 +1.00 +1.10 +1.20 +1.20 +1.20 +1.20   ← 符号相反
```

画成条形图更直观（横轴 ratio，纵轴损失）：

```
  A = +1（这条答案比平均好）

  ratio    损失
   0.8   -0.80  ████████████
   0.9   -0.90  █████████████
   1.0   -1.00  ██████████████
   1.1   -1.10  ████████████████
   1.2   -1.20  ██████████████████  ┐
   1.3   -1.20  ██████████████████  ├ 裁剪平台：不再变
   1.4   -1.20  ██████████████████  │
   1.5   -1.20  ██████████████████  ┘

  A = -1（这条答案比平均差）

  ratio    损失
   0.8   +0.80  ████
   0.9   +0.90  █████
   1.0   +1.00  ██████
   1.1   +1.10  ███████
   1.2   +1.20  ████████  ┐
   1.3   +1.20  ████████  ├ 同样封顶
   1.4   +1.20  ████████  │
   1.5   +1.20  ████████  ┘
```

读图要点：

- **`A = +1` 时**：`ratio` 从 `0.8` 涨到 `1.2`，损失线性下降（概率涨得越多、罚得越少）；`1.2` 之后走平 —— 再涨也不额外奖励，这就是「一步别走太远」。
- **`A = -1` 时**：符号反过来，损失是正的（要增大），同样在 `1.2` 封顶。
- **`ratio < 0.8` 时**：一样的封顶逻辑（下界），只是这两个例子里没画到。
- 这条「平台」正是 PPO 的 clip：`min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)` 在两端各截一刀。

### 7.5 最后怎么变成一个数

```
  L = - 1/(G·T) · Σ_i Σ_t  min( ratio_{i,t} · A_i ,  clip(ratio_{i,t}, 1-ε, 1+ε) · A_i )
      + β · KL(π_θ ‖ π_ref)
```

| 符号 | 含义 | 本教程取值 |
|---|---|---|
| `G` | 每个 prompt 采几条 | `num_generations=8` |
| `T` | 每条 completion 的 token 数 | ≤ `max_completion_length` |
| `ratio` | `exp(logp_new - logp_old)` | 由 trainer 内部算 |
| `ε` | 裁剪幅度 | 默认 0.2（即 0.8~1.2） |
| `A_i` | 组内优势 | 见 7.2 |
| `β` | 参考模型 KL 惩罚 | **默认 0.0**（不用参考模型，省一份显存） |

**只对 completion 的 token 求平均**：prompt 段的 `loss_mask = 0`，不参与。这点和第 1 章的 SFT 掩码完全一致 —— 模型不为「别人写的东西」负责。

### 7.6 一句话记住

> 损失 = 「这条答案比平均好多少」×「这个 token 的概率涨了多少」，涨过头就截断。

---

## 8. 代码与参数

`code/train_grpo.py` 的六个阶段：读数据 → 模板 → 采样打分 → 算优势 → 训练 → 前后对比。

```python
args = GRPOConfig(
    num_generations=8,          # ★ G：每个 prompt 采几条（要能整除有效 batch）
    temperature=1.0,            # ★ 采样温度，太低 → 8 条几乎一样，组内 std≈0
    max_completion_length=256,  # ★ 单条答案 token 上限
    beta=0.0,                   # ★ 0 = 不挂参考模型 KL
    loss_type="dapo",           # ★ TRL 1.16 默认
    scale_rewards="group",      # ★ 组内归一化，即 7.2 的公式
    remove_unused_columns=False,# 奖励函数要读 ground_truth 列
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[format_reward, accuracy_reward],   # 多条奖励会相加
    args=args, train_dataset=ds,
)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `num_generations` | 8 | G。必须整除「有效 batch = batch × 累积 × 卡数」 |
| `temperature` | 1.0 | 太低组内无差异，太高全是废话 |
| `beta` | 0.0 | >0 时挂参考模型 KL（多一份显存） |
| `loss_type` | `"dapo"` | 还有 `"grpo"`、`"bnpo"` 等变体 |
| `scale_rewards` | `"group"` | `"batch"` 则跨 prompt 归一化 |
| `reward_weights` | — | 多条奖励的权重（只作用于 `reward_funcs`） |
| `mask_truncated_completions` | False | 被截断的答案不计入损失 |

---

## 9. 跑起来

脚本：[`code/train_grpo.py`](code/train_grpo.py)（约 640 行）

### 9.1 两套数据，别混用

| | 本地解剖样本 | 官方训练集 |
|---|---|---|
| 位置 | `data/grpo_prompts.jsonl`（3 道题） | `--dataset trl-lib/DeepMath-103K`（10 万道可验证数学题） |
| 用来看 | 模板怎么套、一组答案怎么打分、优势怎么算（第 6 节） | **训练效果**：held-out pass@1 有没有涨 |
| 能不能训出东西 | 不能。3 道题 × 8 条采样，模型早就全对了，`std=0`，一点梯度都没有 | 能 |

这道题最容易踩的坑就是「拿 3 道题训 GRPO」：组内全对 → 优势全 0 → loss 恒等于 0 → 你会以为脚本坏了。
真正训练一定要用足够多、难度有梯度的题（DeepMath 里既有送分题也有竞赛题）。

### 9.2 命令

```bash
cd 03-强化学习-GRPO/code

# 只跑数据链路（只要 tokenizer，不需 GPU）
python train_grpo.py --inspect

# 不加载模型，把第 7 节的数字手算打一遍
python train_grpo.py --explain-loss

# 本地 3 道题冒烟（验证管线；大概率 std=0，这是正常的）
python train_grpo.py --max-steps 2 --num-generations 4 --max-new 128

# 真训练：官方题目 3000 道，留 100 道做 held-out pass@1 评测
python train_grpo.py --dataset trl-lib/DeepMath-103K --limit 3000 --eval-size 100

# 显存紧 / 想快：G=4、短答案
python train_grpo.py --dataset trl-lib/DeepMath-103K --limit 3000 --num-generations 4 --max-new 192
```

依赖：`pip install "trl>=1.16.0" transformers datasets accelerate`

分卡跑（大模型时）：`accelerate launch train_grpo.py`，`num_generations` 要能被卡数整除。

### 9.3 怎么判断真的学会了

| 指标 | 从哪来 | 健康 |
|---|---|---|
| **held-out pass@1** | 100 道没参与训练的题，训练前后各测一次 | 涨 10 个点以上（如 22% → 41%） |
| **格式合规率** | 同上，判 `<think>…</think><answer>…</answer>` 是否完整 | 训练后接近 100% |
| **reward 曲线** | 训练日志 | 缓慢上升，`reward_std` 保持在 0.2–0.7 |

只看 reward 不靠谱：reward 涨了但 pass@1 不动，多半是模型学会了**骗奖励**（比如总是输出一种能拿格式分的模板），这正是下一节要看的坑。

---

## 10. 产物与验收

跑完看 `runs/grpo/report.md`：

| 该看到的 | 正常的样子 | 不正常 |
|---|---|---|
| 优势表 | 同组内有正有负 | 全是 0 → 组内全对/全错 |
| `reward` 曲线 | 缓慢上升 | 一直不动 / 忽然飙到 1 |
| `reward_std` | 0.2~0.7 | ≈0 → 采样没差异；≈1.5 → 全对全错各一半 |
| `kl` | `beta=0` 时无此项 | 突然飙高 → 学崩了 |
| `completions/clipped_ratio` | < 0.3 | 接近 1 → 预算太小，答案全被截断 |

**验收标准**：held-out pass@1 比训练前高 10 个点以上，且格式合规率接近 100%。

---

## 11. 常见坑

1. **`num_generations` 不能整除有效 batch** → 报错或静默丢弃，改成 4/8 这类因子。
2. **组内全对/全错** → `std=0`，`A` 全 0，白跑。降难度或加 `temperature`。
3. **奖励函数返回单个 float** → 所有答案同分，等于没奖励。
4. **忘了 `remove_unused_columns=False`** → `ground_truth` 被丢掉，奖励函数拿不到标准答案。
5. **`max_completion_length` 太小** → 答案被截断，`<answer>` 永远闭合不了，格式分恒 0。
6. **格式奖励和正确性奖励没区分开** → 模型学会「写空标签」骗分（reward hacking）。
7. **`beta>0` 却忘了参考模型显存** → OOM，等于同时装两份模型。
8. **用 `temperature=0.1` 采样 8 条** → 8 条几乎一样，组内没信号。

---

## 12. 练习与自测

1. 把 `format_reward` 改成「`<answer>` 里必须是纯数字」，跑一下，看格式分为什么先升后平。
2. 手算：一组奖励 `[2, 2, 1, 0]`，求 `A`。（答案：μ=1.25，σ=0.829，A≈[0.906, 0.906, -0.302, -1.509]）
3. 把 `scale_rewards` 改成 `"batch"`，观察 `reward` 曲线有什么不同。
4. 故意把 `num_generations` 设成 6（batch 8 不整除），看报错长什么样。

**自测**（能答上来就算过）：

- 为什么 GRPO 不需要 critic 模型？
- `A_i` 里为什么要除标准差？
- 一组全对时，梯度为什么是 0？
- `ratio` 涨到 1.5 时，为什么损失不再变化？

---

下一章讲「环境」：奖励函数只有一次机会、一个分数；环境可以让模型**动手多轮**，每一步都看到结果。
