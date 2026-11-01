"""
手算一遍：SFT 的损失到底是啥。

不加载模型、不需要 GPU，纯数学 —— 因为「交叉熵」这四个字拦住了太多人，
把它摊成 -ln(p) 的几个数字，比任何解释都直观。
"""

from __future__ import annotations

import math

from common import hr


def explain_loss() -> None:
    """不加载模型，把 token 级交叉熵一项一项算出来。"""
    hr("手算一遍：SFT 的损失到底是什么（交叉熵）")
    print("  模型在每个位置输出一整个词表的分数（logits），softmax 之后是这个 token 的概率。")
    print("  损失 = 正确 token 的 -ln(概率)。猜得越准，损失越小。")
    print()
    print(f"  {'正确 token 的概率':>16}  {'-ln(p)':>8}   直觉")
    print("  " + "-" * 56)
    for p, note in [
        (0.99, "几乎确定 → 几乎不罚"),
        (0.90, "很确定"),
        (0.50, "在犹豫"),
        (0.20, "猜错了"),
        (0.01, "完全跑偏 → 罚得很重"),
    ]:
        print(f"  {p:>16.2f}  {-math.log(p):>8.3f}   {note}")
    print()
    print("  一条轨迹里 5 个被监督的 token，假设模型给它们这些概率：")
    print()
    ps = [0.98, 0.95, 0.90, 0.80, 0.50]
    for i, p in enumerate(ps):
        print(f"    token {i + 1}: p = {p:.2f}   →   -ln(p) = {-math.log(p):.3f}")
    mean = sum(-math.log(p) for p in ps) / len(ps)
    print(f"    平均 = {mean:.3f}   ← 这就是 loss（这个 batch 里所有被监督 token 的平均）")
    print()
    print("  关键三点：")
    print("   1) 只有 label != -100 的位置参与 —— prompt 和 tool 结果不算（第 5 节的掩码条）。")
    print("   2) 位置 t 的预测要和位置 t+1 的 token 比（右移一位），不是和它自己比。")
    print("   3) 损失对所有被监督 token 取平均，所以长轨迹里的短片段会被稀释。")
