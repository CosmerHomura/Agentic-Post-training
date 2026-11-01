"""
阶段 2~4：模板 → token → label/mask → batch。

这是全教程最容易「看起来懂了其实没懂」的地方，所以每个阶段都打印出来。
核心事实：喂给模型的不是 JSON，是一段纯文本；模型学的也不是每个 token，
只是 assistant 说的那些 token —— 其它位置在 labels 里被换成 -100。
"""

from __future__ import annotations

from common import RUN_DIR, hr, kv, piece


def patch_template(tpl: str) -> tuple[str, bool]:
    """
    给 chat template 插入 {% generation %} / {% endgeneration %}。

    为什么需要：transformers 只有看到这对标记，才知道「哪几个 token 是模型该生成的」，
    才会返回 assistant_masks。Qwen3 的模板里没有这对标记，所以直接调
    return_assistant_tokens_mask=True 会得到全 0（并给一条 warning）。
    TRL 的 assistant_only_loss=True 在内部做的也是同一件事：打补丁 → 取掩码。
    """
    if "{% generation %}" in tpl or "{%- generation %}" in tpl:
        return tpl, False
    patched = tpl.replace(
        '{%- elif message.role == "assistant" %}',
        '{%- elif message.role == "assistant" %}{%- generation %}',
    )
    patched = patched.replace(
        '{%- elif message.role == "tool" %}',
        '{%- endgeneration %}{%- elif message.role == "tool" %}',
    )
    if "{%- generation %}" not in patched:
        raise RuntimeError("模板结构不认识，没法自动打 generation 补丁；请手动检查 chat_template。")
    return patched, True


def build_masks(tok, messages, tools):
    """
    返回 (input_ids, assistant_mask, 说明)。

    规则（与 TRL 的 assistant_only_loss 同源）：
      - assistant 轮：监督（含它的 <tool_call> JSON 和结尾的 <|im_end|>）
      - user / system / tool 轮：不监督
    另外把每个 assistant 轮开头的 <|im_start|>assistant\\n 三个 token 排除，
    因为那是「给定上下文」，不是模型要说的话。
    """
    tpl, was_patched = patch_template(tok.chat_template)
    old = tok.chat_template
    tok.chat_template = tpl
    try:
        out = tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        ids = list(out["input_ids"])
        mask = list(out["assistant_masks"])
    finally:
        tok.chat_template = old

    if sum(mask) == 0:
        raise RuntimeError("掩码全 0：模板补丁没生效，检查 chat_template 结构。")

    # 去掉每段连续「1」最前面的 3 个 token（<|im_start|> / assistant / 换行）
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            for k in range(i, min(i + 3, j)):
                mask[k] = 0
            i = j
        else:
            i += 1

    note = "模板自带 generation 标记" if not was_patched else "已给模板打 generation 补丁"
    return ids, mask, note


def stage_template(tok, rec):
    hr("阶段 2 / 4   套 chat template（JSON 结构 → 一段纯文本）")
    msgs, tools = rec["messages"], rec.get("tools")
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
    print(f"  渲染结果：{len(text)} 个字符")
    print("  下面原样打印；每行开头的 <|im_start|>role 是模板插的分隔符：")
    print("  " + "-" * 74)
    for line in text.splitlines():
        mark = "  "
        if line.startswith("<|im_start|>"):
            mark = "▸ "
        elif line.startswith("<|im_end|>"):
            mark = "◂ "
        print(f"  {mark}{line}")
    print("  " + "-" * 74)
    return text


def stage_tokens(tok, rec, show_window=14):
    hr("阶段 3 / 4   tokenize → 打掩码 → labels")
    msgs, tools = rec["messages"], rec.get("tools")
    ids, mask, note = build_masks(tok, msgs, tools)
    labels = [t if m else -100 for t, m in zip(ids, mask)]

    n_sup = sum(mask)
    print(f"  {note}")
    print(f"  总 token：{len(ids)}    需要学的 token：{n_sup}（{n_sup / len(ids):.1%}）")
    print()

    # 掩码条：一个字符一个 token
    print("  掩码条（█ = 要学，· = 不学，每行 100 个 token）：")
    for base in range(0, len(mask), 100):
        chunk = mask[base : base + 100]
        print(f"    {base:>4} " + "".join("█" if m else "·" for m in chunk))
    print()

    # 表格：开头一段 + 第一处「开始学」的位置附近 + 结尾一段
    head = list(range(min(10, len(ids))))
    first = mask.index(1) if 1 in mask else 0
    mid = list(range(max(0, first - 6), min(len(ids), first + show_window + 2)))
    tail = list(range(max(0, len(ids) - 10), len(ids)))
    shown = sorted(set(head + mid + tail))
    print(f"  {'idx':>4}  {'token_id':>8}  {'label':>8}  {'学':<2}  token")
    print("  " + "-" * 58)
    prev = None
    for i in shown:
        if prev is not None and i != prev + 1:
            print(f"  {'…':>4}  {'…':>8}  {'…':>8}     …")
        flag = "█" if mask[i] else "·"
        print(f"  {i:>4}  {ids[i]:>8}  {labels[i]:>8}  {flag:<2}  {piece(tok, ids[i])}")
        prev = i
    print()
    print("  上面这一段就是「同一个 token 在第 1 行是原文、在第 3 行变成 -100」的地方：")
    print("  掩码翻转的位置，正好落在 <|im_start|>assistant 之后。")

    full = RUN_DIR / "tokens.txt"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(
        "\n".join(
            f"{i:>5}  {t:>8}  {l:>8}  {'1' if m else '0'}  {piece(tok, t)}"
            for i, (t, l, m) in enumerate(zip(ids, labels, mask))
        ),
        encoding="utf-8",
    )
    print(f"\n  全量表已写入：{full}")
    return ids, mask, labels


def stage_batch(tok, records):
    hr("阶段 4 / 4   拼成 batch（模型真正看到的东西）")
    batch_ids, batch_lab, batch_am = [], [], []
    for rec in records:
        ids, mask, _ = build_masks(tok, rec["messages"], rec.get("tools"))
        batch_ids.append(ids)
        batch_lab.append([t if m else -100 for t, m in zip(ids, mask)])
        batch_am.append([1] * len(ids))

    maxlen = max(len(x) for x in batch_ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for ids, lab, am in zip(batch_ids, batch_lab, batch_am):
        pad_n = maxlen - len(ids)
        ids += [pad] * pad_n
        lab += [-100] * pad_n     # padding 位置必须 -100，否则会去学 pad
        am += [0] * pad_n

    print(f"  样本数：{len(batch_ids)}   序列长度（pad 后）：{maxlen}")
    print()
    shape = f"[{len(batch_ids)}, {maxlen}]"
    kv(
        input_ids=f"{shape}  int64   # pad={pad}",
        attention_mask=f"{shape}  int64   # pad 处为 0",
        labels=f"{shape}  int64   # 不学的位置 = -100",
    )
    print()
    print("  第 1 条样本的头 12 个位置：")
    print(f"  {'idx':>4}  {'input_ids':>10}  {'labels':>10}  {'attn':>4}")
    for i in range(min(12, maxlen)):
        print(f"  {i:>4}  {batch_ids[0][i]:>10}  {batch_lab[0][i]:>10}  {batch_am[0][i]:>4}")
    print()
    print("  注意：labels 只是把不该学的位置换成 -100（CrossEntropy 的 ignore_index），")
    print("        其余位置和 input_ids 完全相同 —— 这就是「右移一位」的自回归目标。")
    try:
        import torch

        t = torch.tensor(batch_ids, dtype=torch.long)
        print(f"\n  torch 版：{tuple(t.shape)}  {t.dtype}  设备无关，训练时由 Trainer 搬到 GPU")
    except ImportError:
        print("\n  （没装 torch，上面用纯 Python 列表演示形状；训练阶段需要 torch）")
