"""
阶段 4~5 + held-out 探针：学生的尝试，以及「教师会怎么写」。

on-policy 的关键动作在这里：先让学生自己写（写好写坏都行），
再拿同一串 token 喂给教师，看两边在「下一个词」上的分布差多远。
"""

from __future__ import annotations

from common import device_of, dtype_of, hr, ids_of, piece, tid_of


def stage_generate(prompt_ids, tok, student_name, gen_tokens: int, temperature: float, seed: int = 0):
    """学生写 completion —— on-policy 的那一步。"""
    hr("阶段 4 / 5   学生自己生成 completion（on-policy）")
    import torch
    from transformers import AutoModelForCausalLM

    dev, dtype = device_of(torch), dtype_of(torch)
    torch.manual_seed(seed)
    student = AutoModelForCausalLM.from_pretrained(student_name, dtype=dtype).to(dev).eval()

    ids = torch.tensor([prompt_ids], device=dev)
    with torch.no_grad():
        out = student.generate(
            ids,
            max_new_tokens=gen_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen = out[0][len(prompt_ids):].tolist()
    text = tok.decode(gen, skip_special_tokens=False)

    print(f"  学生：{student_name}")
    print(f"  采样：temperature={temperature}  max_new_tokens={gen_tokens}")
    print()
    print(f"  生成 {len(gen)} 个 token：")
    print("  " + "-" * 74)
    for line in text.split("\n"):
        print("  " + line)
    print("  " + "-" * 74)
    print()
    print("  这段文字**没有标准答案可以对照**。它是学生自己的尝试，可能对、可能错、")
    print("  可能中途跑偏 —— 教师不改它，只是告诉它「换了我，每个位置会怎么分布」。")
    print()
    print("  completion 的 token 表（前 24 个）：")
    print(f"  {'i':>3}  {'token_id':>8}  token")
    print("  " + "-" * 40)
    for i, tid in enumerate(gen[:24]):
        print(f"  {i:>3}  {tid:>8}  {piece(tok, tid)}")
    if len(gen) > 24:
        print(f"  …  还有 {len(gen) - 24} 个")
    return student, gen


def stage_distributions(prompt_ids, gen, tok, student, student_name, teacher_name,
                        beta: float, temperature: float):
    """
    教师 vs 学生：同一个上下文，各自的 next-token 分布。

    p_T = softmax(teacher_logits / T)，p_S = softmax(student_logits / T)
    p_M = (1-β)·p_S + β·p_T
    L   = β·KL(p_T‖p_M) + (1-β)·KL(p_S‖p_M)
    """
    hr("阶段 5 / 5   教师 vs 学生：逐 token 的分布差多少")
    import torch
    from transformers import AutoModelForCausalLM

    dev, dtype = device_of(torch), dtype_of(torch)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, dtype=dtype).to(dev).eval()

    full = prompt_ids + gen
    t = torch.tensor([full], device=dev)
    with torch.no_grad():
        ls = student(t).logits[0]
        lt = teacher(t).logits[0]

    # 第 i 个 completion token 的预测，来自位置 len(prompt)+i-1
    start = len(prompt_ids) - 1
    tmp = temperature

    def softmax(x):
        x = x / tmp
        return torch.softmax(x, dim=-1)

    rows = []
    for i, tid in enumerate(gen):
        pos = start + i
        ps, pt = softmax(ls[pos]), softmax(lt[pos])
        p_tok_s, p_tok_t = float(ps[tid]), float(pt[tid])
        top_t = torch.topk(pt, 3)
        kl_fwd = float(torch.sum(pt * (torch.log(pt + 1e-9) - torch.log(ps + 1e-9))))
        rows.append(
            {
                "i": i,
                "token": tok.decode([tid], skip_special_tokens=False),
                "p_student": p_tok_s,
                "p_teacher": p_tok_t,
                "kl_forward": kl_fwd,
                "teacher_top1": tok.decode([int(top_t.indices[0])], skip_special_tokens=False),
                "teacher_top1_p": float(top_t.values[0]),
            }
        )

    print(f"  教师：{teacher_name}（{sum(p.numel() for p in teacher.parameters()) / 1e6:.0f}M 参数）")
    print(f"  学生：{student_name}（{sum(p.numel() for p in student.parameters()) / 1e6:.0f}M 参数）")
    print()
    print("  同一个上下文下，两边对「下一个词」的判断：")
    print(f"  {'i':>3}  {'学生写的词':<12} {'p_学生':>8} {'p_教师':>8} {'教师首选':<12} {'p_首选':>7}")
    print("  " + "-" * 68)
    for r in rows[:14]:
        print(
            f"  {r['i']:>3}  {piece(tok, tid_of(tok, r['token']))!s:<12} "
            f"{r['p_student']:>8.3f} {r['p_teacher']:>8.3f} "
            f"{piece(tok, tid_of(tok, r['teacher_top1']))!s:<12} {r['teacher_top1_p']:>7.3f}"
        )
    if len(rows) > 14:
        print(f"  …  还有 {len(rows) - 14} 个 token")

    mean_kl = sum(r["kl_forward"] for r in rows) / len(rows)
    # 用 β 混合出 loss（与 TRL 的 GKD 目标一致）
    loss_terms = []
    for i, tid in enumerate(gen):
        pos = start + i
        ps, pt = softmax(ls[pos]), softmax(lt[pos])
        pm = (1 - beta) * ps + beta * pt
        kl_tm = torch.sum(pt * (torch.log(pt + 1e-9) - torch.log(pm + 1e-9)))
        kl_sm = torch.sum(ps * (torch.log(ps + 1e-9) - torch.log(pm + 1e-9)))
        loss_terms.append(float(beta * kl_tm + (1 - beta) * kl_sm))
    loss = sum(loss_terms) / len(loss_terms)

    print()
    print(f"  平均前向 KL  D_KL(p_T‖p_S) = {mean_kl:.4f}   （越高说明学生离教师越远）")
    print(f"  β = {beta} 时的蒸馏损失        = {loss:.4f}")
    print()
    print("  β 怎么选（同一个公式的三种含义）：")
    print("    β = 0.0  →  L = KL(p_T‖p_S)   前向 KL：学生要覆盖教师的**所有**可能性，容易学到「模糊话」")
    print("    β = 1.0  →  L = KL(p_S‖p_T)   反向 KL：学生只盯教师最确定的那条路，容易「塌缩」（TRL 默认）")
    print("    β = 0.5  →  L = JSD            折中，也是本脚本默认")
    return rows, mean_kl, loss, teacher


def probe_kl(tok, student, teacher, eval_rows, gen_tokens: int, temperature: float, k: int = 8):
    """
    held-out 上量一次「学生离教师有多远」：越大说明还没学会教师。

    做法：学生自己生成 → 教师在同一串 token 上给分布 → 算平均前向 KL。
    蒸馏做对了，这个数应该明显下降。
    """
    import torch

    dev = student.device
    teacher = teacher.to(dev).eval()
    kl_sum, n_tok, samples = 0.0, 0, []
    for idx, r in enumerate(eval_rows[:k]):
        ids = ids_of(tok, r["prompt"], add_generation_prompt=True)
        t = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = student.generate(
                t, max_new_tokens=gen_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            gen = out[0][len(ids):].tolist()
            if not gen:
                continue
            full = torch.tensor([ids + gen], device=dev)
            ls = student(full).logits[0]
            lt = teacher(full).logits[0]
        start = len(ids) - 1
        for i in range(len(gen)):
            pos = start + i
            ps = torch.softmax(ls[pos] / temperature, dim=-1)
            pt = torch.softmax(lt[pos] / temperature, dim=-1)
            kl_sum += float(torch.sum(pt * (torch.log(pt + 1e-9) - torch.log(ps + 1e-9))))
            n_tok += 1
        if idx < 2:
            samples.append(
                {"prompt": r["prompt"][-1]["content"], "out": tok.decode(gen, skip_special_tokens=False)}
            )
    return (kl_sum / n_tok if n_tok else float("nan")), samples
