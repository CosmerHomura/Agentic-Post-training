"""
阶段 3 · 采样与打分（打印版）+ held-out pass@1。

这一步是 SFT 与 GRPO 的分叉点：
    SFT  往 prompt 后面塞一条**人写的**轨迹
    GRPO 让模型**自己**写 G 条，然后组内比较
"""

from __future__ import annotations

from common import hr, ids_of, piece
from grpo_data import ANSWER_RE, FORMAT_RE, norm_ans
from grpo_rewards import accuracy_reward, format_reward


def stage_template(rows, tok) -> list[int]:
    hr("阶段 2 / 5   套模板 + tokenize（和 SFT 前半段完全一样）")
    msgs = rows[0]["prompt"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"  渲染结果：{len(text)} 个字符")
    print("  " + "-" * 74)
    for line in text.split("\n"):
        print("  " + line.replace("<|im_start|>", "▸ <|im_start|>"))
    print("  " + "-" * 74)

    ids = ids_of(tok, msgs, add_generation_prompt=True)
    print(f"\n  prompt token 数：{len(ids)}；最后三个：{[piece(tok, t) for t in ids[-3:]]}")
    print()
    print("  到这里和 SFT 一模一样。**差别从下一步开始**：")
    print("  SFT 往这个位置塞一条人写的轨迹；GRPO 让模型自己写 G 条。")
    return ids


def stage_sample_and_score(rows, tok, model_name: str, num_gen: int, temperature: float, max_new: int):
    hr(f"阶段 3 / 5   同一个 prompt 采 {num_gen} 条，然后逐条打分")
    import torch
    from transformers import AutoModelForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(dev).eval()

    row = rows[0]
    ids = ids_of(tok, row["prompt"], add_generation_prompt=True)
    t = torch.tensor([ids], device=dev)

    print(f"  模型：{model_name}    temperature={temperature}    G={num_gen}")
    print("  注意：这 G 条是**各自独立采样**的，同一条 prompt 会有对有错 —— 这正是 GRPO 需要的。")
    print()

    comps = []
    for i in range(num_gen):
        with torch.no_grad():
            out = model.generate(
                t, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0][len(ids):], skip_special_tokens=True)
        comps.append([{"role": "assistant", "content": text}])
        print(f"  ── 第 {i + 1} 条 ──")
        print("      " + text.replace("\n", "\n      ")[:400])
        print()

    f_scores = format_reward(comps)
    a_scores = accuracy_reward(comps, ground_truth=[row["ground_truth"]] * num_gen)
    total = [f + a for f, a in zip(f_scores, a_scores)]

    mean = sum(total) / len(total)
    std = (sum((x - mean) ** 2 for x in total) / len(total)) ** 0.5
    adv = [(x - mean) / (std + 1e-4) for x in total]

    print()
    print("  打分汇总（reward = format + accuracy）：")
    print(f"  {'i':>3}  {'format':>7}  {'accuracy':>9}  {'reward':>7}")
    print("  " + "-" * 36)
    for i, (f, a) in enumerate(zip(f_scores, a_scores)):
        print(f"  {i + 1:>3}  {f:>7.1f}  {a:>9.1f}  {f + a:>7.1f}")

    print()
    print(f"  组内均值 mean = {mean:.3f}   标准差 std = {std:.3f}")
    print()
    print("  优势 A_i = (r_i - mean) / (std + eps)：")
    print(f"  {'i':>3}  {'reward':>7}  {'A_i':>8}  含义")
    print("  " + "-" * 52)
    for i, (r, a) in enumerate(zip(total, adv)):
        tag = "比平均好 → 提高它的概率" if a > 0.05 else ("比平均差 → 压低" if a < -0.05 else "≈平均 → 不动")
        print(f"  {i + 1:>3}  {r:>7.1f}  {a:>8.3f}  {tag}")

    if std < 1e-3:
        print()
        print("  ⚠️ std ≈ 0：这一组全对或全错，A 全是 0，这一步学不到任何东西。")
        print("     这就是为什么「任务太难或太简单」都训不动，要选模型能做对一部分的题。")

    return {"f_scores": f_scores, "a_scores": a_scores, "total": total,
            "mean": mean, "std": std, "adv": adv}, comps


def pass_at_1(tok, model, eval_rows, max_new: int, temperature: float = 0.7):
    """held-out 上每题采 1 条判对错 —— 这才是「有没有学会」的硬指标。

    loss 降了但 pass@1 没动，说明模型只是变得更自信，不是变得更对。
    """
    import torch

    dev = model.device
    ok = fmt = 0
    samples = []
    for r in eval_rows:
        ids = ids_of(tok, r["prompt"], add_generation_prompt=True)
        t = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model.generate(
                t, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0][len(ids):], skip_special_tokens=True)
        m = ANSWER_RE.search(text)
        got = norm_ans(m.group(1)) if m else ""
        ok += int(bool(got) and got == norm_ans(r["ground_truth"]))
        fmt += int(bool(FORMAT_RE.match(text.strip())))
        if len(samples) < 2:
            samples.append({"gt": r["ground_truth"], "got": got, "out": text.strip()[:400]})
    n = max(1, len(eval_rows))
    return {"pass1": ok / n, "fmt": fmt / n, "n": len(eval_rows), "samples": samples}
