"""
手算一遍：一个位置的蒸馏损失。

词表砍到 4 个词，KL / JSD 逐项算出来 —— 因为「前向 KL 和反向 KL 不一样」
这件事，看公式永远记不住，看一张逐词贡献表才记得住。
"""

from __future__ import annotations

import math

from common import hr


def explain_loss(beta: float) -> None:
    """词表砍到 4 个词，把 KL / JSD 逐项算出来（纯数学，不加载任何模型）。"""
    words = ["先", "看", "<tool_call>", '{"']
    p_t = [0.60, 0.30, 0.08, 0.02]   # 教师
    p_s = [0.50, 0.30, 0.15, 0.05]   # 学生：把 0.10 从 <tool_call> 挪给了 {

    def kl(a, b):
        return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0)

    hr("手算一遍：一个位置的蒸馏损失（词表只有 4 个词）")
    print("  同一个位置，教师和学生的完整分布：")
    print()
    print(f"  {'词':<12} {'p_T':>6} {'p_S':>6}   p_T - p_S")
    print("  " + "-" * 44)
    for w, t, s in zip(words, p_t, p_s):
        flag = "  ← 学生低估" if t - s > 1e-9 else ("  ← 学生高估" if s - t > 1e-9 else "")
        print(f"  {w:<12} {t:>6.2f} {s:>6.2f}   {t - s:+.2f}{flag}")

    print("\n  ① 前向 KL  KL(p_T‖p_S) = Σ p_T·ln(p_T/p_S)")
    print(f"  {'词':<12} {'p_T·ln(p_T/p_S)':>16}")
    print("  " + "-" * 32)
    for w, t, s in zip(words, p_t, p_s):
        print(f"  {w:<12} {t * math.log(t / s):>+16.4f}")
    fwd = kl(p_t, p_s)
    print(f"  {'合计':<12} {fwd:>+16.4f}")

    print("\n  ② 反向 KL  KL(p_S‖p_T) = Σ p_S·ln(p_S/p_T)")
    print(f"  {'词':<12} {'p_S·ln(p_S/p_T)':>16}")
    print("  " + "-" * 32)
    for w, t, s in zip(words, p_t, p_s):
        print(f"  {w:<12} {s * math.log(s / t):>+16.4f}")
    rev = kl(p_s, p_t)
    print(f"  {'合计':<12} {rev:>+16.4f}")

    print(f"\n  两个方向不相等（{fwd:.4f} vs {rev:.4f}）—— KL 不对称，这就是 β 的用处。")

    p_m = [beta * t + (1 - beta) * s for t, s in zip(p_t, p_s)]
    l_mix = beta * kl(p_t, p_m) + (1 - beta) * kl(p_s, p_m)
    print(f"\n  ③ 取 β = {beta}：先混合 p_M = β·p_T + (1-β)·p_S")
    print(f"     p_M = [{', '.join(f'{x:.4f}' for x in p_m)}]")
    print(f"     KL(p_T‖p_M) = {kl(p_t, p_m):.4f}")
    print(f"     KL(p_S‖p_M) = {kl(p_s, p_m):.4f}")
    print(f"     中间式 L_mix = β·KL(p_T‖p_M) + (1-β)·KL(p_S‖p_M) = {l_mix:.4f}")

    # 端点上中间式会退化成 0：β=1 时 p_M=p_T 且权重 (1-β)=0；β=0 时同理。
    # TRL 的 generalized_jsd_loss() 因此把两个端点写成独立分支（源码 278-297 行）。
    if beta == 0.0:
        L = kl(p_t, p_s)
        print(f"     β=0 用分支：L = KL(p_T‖p_S) = {L:.4f}  （前向 KL）")
    elif beta == 1.0:
        L = kl(p_s, p_t)
        print(f"     β=1 用分支：L = KL(p_S‖p_T) = {L:.4f}  （反向 KL）")
    else:
        L = l_mix
        if beta < 0.1:
            print(f"     靠近 β=0 端点，中间式是线性的：L_mix ≈ β·KL(p_T‖p_S) = {beta:.2f} × {fwd:.4f}")
        elif beta > 0.9:
            print(f"     靠近 β=1 端点，中间式是线性的：L_mix ≈ (1-β)·KL(p_S‖p_T) = {1 - beta:.2f} × {rev:.4f}")
        else:
            print("     中间区间：p_M 离两边都不远，所以 L 比两条 KL 都小")

    print(f"\n  ⇒ 这次训练用的损失 L = {L:.4f}")
    print("  β = 0 是前向 KL，β = 1 是反向 KL（两个端点由代码分支给出，")
    print("  因为中间式在端点会退化成 0），中间才是广义 JSD。")

    print("\n  ④ 真实训练里：")
    print("     · 每个位置都算一次这样的 L_t（词表 151936 维，不是 4 维）")
    print("     · 只对 completion 的 token 取平均，prompt 段 mask=0")
    print("     · 整个 batch 再平均 → 你看到的 loss（0.0x ~ 0.5 都是正常的）")
