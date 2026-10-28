"""
第 3 步 · 奖励函数
GRPO 里的 reward function 就是"裁判"：输入一批模型输出，输出一批分数。

硬性契约（来自 TRL 文档）：
  1. 用关键字参数接收你想看的字段：prompts / completions / completion_ids / trainer_state ...
     —— 以及数据集里除 prompt 之外的**所有列**（比如 ground_truth）会自动作为 kwargs 传进来。
  2. **返回 list[float]**，长度必须等于这一批 completion 的数量，顺序一一对应。
  3. 可以是同步函数，也可以是 `async def`（多个异步奖励会用 asyncio.gather 并发执行）。

另外：第 5 步的 AsyncGRPOTrainer 会把这些函数 pickle 到另一个进程，
      所以它们必须定义在**模块顶层**（lambda / 闭包会报 TypeError）。
"""

import re


def format_reward(completions, **kwargs):
    """格式分：必须严格是 <think>...</think><answer>...</answer> 的结构。

    completions 在标准格式下是字符串列表，在对话格式下是 message dict 列表，
    所以这里用 completion[0]["content"] 取值。
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    return [
        1.0 if re.match(pattern, completion[0]["content"].strip(), re.DOTALL) else 0.0
        for completion in completions
    ]


def accuracy_reward(completions, ground_truth, **kwargs):
    """答案分：把 <answer> 里的内容抽出来，和标准答案做数值比较。

    ground_truth 来自数据集那一列，是字符串列表（每个 prompt 一个）。
    注意 completions 与 ground_truth 的长度都是"这一批 completion 数"，
    但 ground_truth 是按 prompt 给的常量，TRL 已经替你广播好了。
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        text = completion[0]["content"]
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if m is None:
            rewards.append(0.0)
            continue
        try:
            pred = float(m.group(1).strip())
            rewards.append(1.0 if abs(pred - float(gt)) < 1e-6 else 0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards


def length_penalty(completion_ids, **kwargs):
    """负向奖励示例：太啰嗦要扣分。

    只想"看 token id"时，直接用 completion_ids（每个元素是 token id 列表）。
    返回负值就会被当作减分项，和正奖励相加。
    """
    return [-0.001 * len(ids) for ids in completion_ids]


# 一行验证：默认 8 个 generation 时，每个 reward 会收到 8 个元素
if __name__ == "__main__":
    demo = [[{"content": "<think>23*47=1081</think><answer>1081</answer>"}]]
    print(format_reward(demo))                      # [1.0]
    print(accuracy_reward(demo, ["1081"]))          # [1.0]
    print(length_penalty([[1, 2, 3]]))              # [-0.003]
