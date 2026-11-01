"""
手算一遍：GRPO 的损失到底在算什么。

不加载任何模型 —— 只为了让你在第一次看到 loss 是 0.0003 时不慌：
「好」的那一半可能已经被 clip 封顶，剩下的全靠「差」的那一半在推。
"""

from __future__ import annotations

from common import hr


def explain_loss() -> None:
    hr("手算一遍：GRPO 的损失到底在算什么（不加载模型）")
    print("  假设一次采样 4 条，奖励分别是 1、0、1、0（对/错），G = 4。")
    rewards = [1.0, 0.0, 1.0, 0.0]
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    adv = [(r - mean) / (std + 1e-4) for r in rewards]
    print(f"\n  ① 组内归一化：mean = {mean:.2f}，std = {std:.2f}")
    print(f"     A = {[round(a, 3) for a in adv]}")
    print("     → 对的两条各 +1，错的两条各 -1（正负对称，这是 GRPO 省掉 critic 的关键）")

    print("\n  ② 对某一个 token，看新旧策略的概率比：")
    print("     设某条**正确**答案（A=+1）的某个 token：")
    print("       旧策略 p_old = 0.20，新策略 p_new = 0.26  → ratio = p_new/p_old = 1.30")
    ratio = 0.26 / 0.20
    print(f"       未裁剪项 ratio·A        = {ratio:.2f} × 1 = {ratio:.2f}")
    print("       裁剪项 clip(0.8,1.2)·A  = 1.20 × 1 = 1.20")
    print(f"       损失贡献 = -min({ratio:.2f}, 1.20) = -1.20")
    print("     → 概率涨到 1.2 倍以后就不再奖励了，防止一步走太远")

    print()
    print("     若这条是**错误**答案（A=-1），同样 p_new 从 0.20 涨到 0.26：")
    print("       ratio·A = 1.30 × (-1) = -1.30；clip 后 = 1.20 × (-1) = -1.20")
    print("       损失贡献 = -min(-1.30, -1.20) = +1.30")
    print("     → 错误答案的概率被推高时，损失反而变大，梯度把它压回去")

    print("\n  ③ 最后对这个 batch 里**所有 completion 的 token** 取平均：")
    print("     L = -1/(G·T) Σ_i Σ_t min(ratio·A_i, clip(ratio,1±ε)·A_i) + β·KL")
    print("     注意：prompt 段不参与（mask 掉），只有模型自己生成的 token 算。")
    print("     本教程默认 β = 0.0（不挂参考模型 KL），省一份显存。")

    print("\n  ④ 读你训练日志时的对照：")
    print("     · ratio 全落在 0.8~1.2 内 → 每步都在「奖励平台」上，loss 很小是正常的")
    print("     · loss 趋近 0 不一定是学会，可能只是所有样本的 A 都趋近 0（std 也趋近 0）")
    print("     · 真正要看的是 reward 与 held-out pass@1 有没有一起动")
