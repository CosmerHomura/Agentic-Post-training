"""
奖励函数：GRPO 唯一的监督来源。

契约（TRL 1.16）：
  · 第一个参数是 completions，可再加 **kwargs 拿 prompts / completion_ids /
    数据集里的其他列 / trainer_state ...
  · **返回 list[float]，长度必须等于 completions 的条数**
  · 可以是普通函数，也可以是 async def

这里两个最常用的：格式分（照不照格式）+ 正确性分（答案对不对）。
"""

from __future__ import annotations

import os

from grpo_data import ANSWER_RE, FORMAT_RE, norm_ans

_PRINTED = {"done": False}


def content_of(comp) -> str:
    """对话格式下 completion 是 [{role, content}] 列表，纯字符串格式下就是 str。"""
    return comp[0]["content"] if isinstance(comp, (list, tuple)) else comp


def format_reward(completions, **kwargs):
    """必须 <think>…</think><answer>…</answer>，顺序不能反。"""
    scores = [1.0 if FORMAT_RE.match(content_of(c).strip()) else 0.0 for c in completions]
    _print_once("format", scores)
    return scores


def accuracy_reward(completions, ground_truth=None, **kwargs):
    """从 <answer> 里抠出答案，和 ground_truth 归一化后比较。"""
    gts = ground_truth or [None] * len(completions)
    scores = []
    for comp, gt in zip(completions, gts):
        m = ANSWER_RE.search(content_of(comp))
        got = norm_ans(m.group(1)) if m else ""
        scores.append(1.0 if got and got == norm_ans(gt) else 0.0)
    _print_once("accuracy", scores)
    return scores


def _print_once(which: str, scores) -> None:
    """训练时奖励函数会被调用几百次，只在第一次打印，避免刷屏。

    想关掉：`GRPO_SHOW_REWARDS=0 python 03-.../train_grpo.py`
    """
    if _PRINTED["done"] or os.environ.get("GRPO_SHOW_REWARDS", "1") != "1":
        return
    print(f"  [reward_func] {which} 对 {len(scores)} 条 completion 打分：{scores}")
    if which == "accuracy":
        _PRINTED["done"] = True


def score_group(comps, ground_truth, num_gen: int):
    """把一组 completion 打分 + 组内归一化成优势。

    返回 dict：f_scores / a_scores / total / mean / std / adv。
    """
    msgs = [c if isinstance(c, (list, tuple)) else [{"role": "assistant", "content": c}] for c in comps]
    f_scores = format_reward(msgs)
    a_scores = accuracy_reward(msgs, ground_truth=[ground_truth] * len(msgs))
    total = [f + a for f, a in zip(f_scores, a_scores)]

    mean = sum(total) / len(total)
    var = sum((x - mean) ** 2 for x in total) / len(total)
    std = var ** 0.5
    adv = [(x - mean) / (std + 1e-4) for x in total]
    return {
        "f_scores": f_scores,
        "a_scores": a_scores,
        "total": total,
        "mean": mean,
        "std": std,
        "adv": adv,
        "n": num_gen,
    }
