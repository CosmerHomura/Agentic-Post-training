"""
阶段 2~3：模板与 tokenize。

和 SFT 唯一但关键的差别：这里用 add_generation_prompt=True ——
模板会在结尾留一个 `<|im_start|>assistant\\n`，等模型往下写。
SFT 用的数据里已经有答案，所以不需要这个「留白」。
"""

from __future__ import annotations

from common import hr, ids_of, piece


def stage_template(tok, rows) -> str:
    hr("阶段 2 / 5   套 chat template（JSON → 纯文本）")
    msgs = rows[0]["prompt"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"  渲染结果：{len(text)} 个字符（add_generation_prompt=True，末尾多出 assistant 的角色头）")
    print("  " + "-" * 74)
    for line in text.split("\n"):
        print("  " + line.replace("<|im_start|>", "▸ <|im_start|>"))
    print("  " + "-" * 74)
    print()
    print("  三个关键点：")
    print("   1) 结尾的 ▸ <|im_start|>assistant 是**留白**：它在等模型往下写，不是数据内容。")
    print("   2) 蒸馏没有 <tool_response>、没有 <tool_call> —— 因为数据里没有工具调用示例；")
    print("      想让蒸馏学会多轮，得在 prompt 里带 tools，并用 max_tool_calling_iterations 放开。")
    print("   3) 学生与教师必须用同一个 tokenizer，否则「第 7 个 token」指的不是同一个词。")
    return text


def stage_tokens(tok, rows) -> list[int]:
    hr("阶段 3 / 5   tokenize 后，模型真正看到的 prompt")
    msgs = rows[0]["prompt"]
    ids = ids_of(tok, msgs, add_generation_prompt=True)
    print(f"  prompt token 数：{len(ids)}")
    print()
    print(f"  {'idx':>4}  {'token_id':>8}  token")
    print("  " + "-" * 44)
    for i, tid in enumerate(ids):
        print(f"  {i:>4}  {tid:>8}  {piece(tok, tid)}")
    print()
    print("  最后几个 token 是模板加的「请开始写」：")
    print(f"    {[piece(tok, t) for t in ids[-3:]]}")
    return ids
