# -*- coding: utf-8 -*-
"""
把 8 个 README.md 打包成一个单文件网页：index.html
用法：python _site/build.py
产物：F:\\study\\AI\\Agentic Post-training\\index.html（直接双击打开，可加书签）
"""
import html
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (章节 id, 目录名, 侧边栏短标题, 副标题)
CHAPTERS = [
    ("ch0", ".", "总目录", "怎么用这份教程"),
    ("ch1", "00-开始之前", "00 · 开始之前", "先分清 agent / harness / 环境"),
    ("ch2", "01-监督微调-SFT", "01 · 监督微调 SFT", "把正确轨迹背下来"),
    ("ch3", "02-蒸馏", "02 · 蒸馏", "用教师的概率当监督"),
    ("ch4", "03-强化学习-GRPO", "03 · 强化学习 GRPO", "只有一个分数怎么学"),
    ("ch5", "04-RL环境", "04 · RL 环境", "多轮交互与工具"),
    ("ch6", "05-Harness与规模化", "05 · Harness 与规模化", "真实 agent + 异步训练"),
    ("ch7", "99-附录-资源与术语", "99 · 附录", "原文对照 / 链接 / 术语 / FAQ"),
]

DIR2CH = {d: cid for cid, d, _, _ in CHAPTERS}

# 除了 README.md，还要并入哪些同目录下的 md（按顺序追加，各自的 # 标题会被丢掉）
CHAPTER_EXTRAS = {
    "ch1": ["cloud-setup.md"],
}

NUMS = {"ch0": "目", "ch1": "00", "ch2": "01", "ch3": "02", "ch4": "03", "ch5": "04", "ch6": "05", "ch7": "99"}


# ---------------------------------------------------------------- markdown
def esc(s):
    return html.escape(s, quote=False)


def plain(s):
    """去掉 markdown 标记，给侧边栏用。"""
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\s*\(详见.*?\)$", "", s)
    return s.strip()


def slug(title):
    """近似 GitHub 的锚点算法，用来把 xxx/README.md#锚点 映射到章节内小节。"""
    t = title.strip().lower()
    t = re.sub(r"[^\w\u4e00-\u9fff \-]", "", t)   # 去掉标点（·、（）等）
    t = t.replace(" ", " ")
    t = re.sub(r"\s", "-", t)
    return t


class Ctx:
    """一次渲染需要的上下文：内部链接重写。"""

    def __init__(self, slug_map):
        self.slug_map = slug_map      # (ch, slug) -> section id


def inline(text, ctx):
    codes = []

    def keep(m):
        codes.append(m.group(2))
        return "\x00%d\x00" % (len(codes) - 1)

    text = re.sub(r"(`+)(.+?)\1", keep, text, flags=re.S)
    text = esc(text)
    text = text.replace("\x01", "&#124;")

    # 图片（教程里没用到，保险起见）
    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", r'<img alt="\1" src="\2">', text)

    # 链接
    def link(m):
        label, url = m.group(1), m.group(2)
        url = url.replace("&amp;", "&")
        if not url.startswith(("http://", "https://", "mailto:")):
            url = rewrite_link(url, ctx)
        if url.startswith("#"):
            return '<a class="il" href="%s">%s</a>' % (esc(url), label)
        return '<a href="%s" target="_blank" rel="noopener">%s</a>' % (esc(url), label)

    text = re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", link, text)
    text = re.sub(r"&lt;(https?://[^&\s]+?)&gt;", r'<a href="\1" target="_blank" rel="noopener">\1</a>', text)

    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text, flags=re.S)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)

    def restore(m):
        return "<code>%s</code>" % esc(codes[int(m.group(1))]).replace("\x01", "|")

    return re.sub(r"\x00(\d+)\x00", restore, text)


def rewrite_link(url, ctx):
    """把 ../02-蒸馏/README.md#x 变成 #/ch3/s2"""
    path, _, anchor = url.partition("#")
    path = path.strip()
    if not path or path.startswith("http"):
        return url
    norm = path.replace("\\", "/")
    d = os.path.basename(os.path.dirname(norm)) if "/" in norm else ""
    if norm.endswith("README.md") and not d:
        cid = "ch0"
    elif norm.endswith("README.md"):
        cid = DIR2CH.get(d)
    else:
        cid = None
    if not cid:
        return "#"
    if anchor:
        sid = ctx.slug_map.get((cid, anchor))
        if sid:
            return "#/%s/%s" % (cid, sid)
    return "#/%s" % cid


def is_table_start(lines, i):
    if i + 1 >= len(lines):
        return False
    a, b = lines[i], lines[i + 1]
    if "|" not in a:
        return False
    return bool(re.match(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$", b) and "-" in b)


def split_row(row):
    """按 | 切单元格，但跳过 `code span` 里的竖线和转义的 \\|。"""
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    cells, cur, in_code, i = [], [], False, 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row) and row[i + 1] == "|":
            cur.append("\x01")
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
            cur.append(ch)
            i += 1
            continue
        if ch == "|" and not in_code:
            cells.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur))
    return [c.strip() for c in cells]


def render_blocks(lines, ctx, sec_id):
    out = []
    i, n = 0, len(lines)
    hn = 0
    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        # 代码块
        m = re.match(r"^```(.*)$", s)
        if m:
            lang = m.group(1).strip()
            i += 1
            buf = []
            while i < n and not re.match(r"^```\s*$", lines[i].strip()):
                buf.append(lines[i])
                i += 1
            i += 1
            code = "\n".join(buf)
            cls = ' class="lang-%s"' % esc(lang) if lang else ""
            out.append('<div class="codewrap"><pre><code%s>%s</code></pre></div>' % (cls, esc(code)))
            continue

        # 分隔线
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", s):
            out.append('<hr>')
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            level = len(m.group(1))
            txt = m.group(2).strip()
            if level >= 2:
                hn += 1
                hid = "%s-h%d" % (sec_id, hn)
                out.append('<h%d id="%s">%s</h%d>' % (level, hid, inline(txt, ctx), level))
            else:
                out.append('<h%d>%s</h%d>' % (level, inline(txt, ctx), level))
            i += 1
            continue

        # 表格
        if is_table_start(lines, i):
            head = split_row(lines[i])
            i += 2
            rows = []
            while i < n and lines[i].strip() and "|" in lines[i]:
                rows.append(split_row(lines[i]))
                i += 1
            t = ['<div class="tablewrap"><table><thead><tr>']
            for c in head:
                t.append("<th>%s</th>" % inline(c, ctx))
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>")
                for k, c in enumerate(r):
                    cls = ' class="num"' if re.fullmatch(r"-?\d+(\.\d+)?", c) else ""
                    t.append("<td%s>%s</td>" % (cls, inline(c, ctx)))
                t.append("</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue

        # 引用
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            inner = render_blocks(buf, ctx, sec_id)
            out.append('<blockquote>%s</blockquote>' % inner)
            continue

        # 列表
        if re.match(r"^([-*+]|\d+\.)\s+", s):
            ordered = bool(re.match(r"^\d+\.\s+", s))
            items = []
            while i < n:
                cur = lines[i]
                mm = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", cur)
                if mm:
                    items.append(mm.group(3))
                    i += 1
                elif cur.strip() and cur.startswith(("  ", "\t")) and items:
                    items[-1] += " " + cur.strip()
                    i += 1
                else:
                    break
            tag = "ol" if ordered else "ul"
            li = "".join("<li>%s</li>" % inline(x, ctx) for x in items)
            out.append("<%s>%s</%s>" % (tag, li, tag))
            continue

        # 段落
        buf = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break
            if re.match(r"^(```|#{1,6}\s|>|\s*([-*+]|\d+\.)\s)", nxt):
                break
            if is_table_start(lines, i):
                break
            buf.append(nxt)
            i += 1
        out.append("<p>%s</p>" % inline(" ".join(x.strip() for x in buf), ctx))

    return "\n".join(out)


NAV_RE = re.compile(r"^\[(←|→|\u2190|\u2192)")


def strip_nav(lines):
    keep = []
    for ln in lines:
        s = ln.strip()
        if s and len(s) < 500 and "](" in s and re.search(r"(上一节|下一节|总目录|返回)", s):
            continue
        if NAV_RE.match(s):
            continue
        keep.append(ln)
    return keep


def build_chapter(cid, folder, title, subtitle, slug_map):
    path = os.path.join(ROOT, folder, "README.md")
    text = io.open(path, encoding="utf-8").read()

    # 并入同目录的附加 md：只取其第一个 ## 起的内容（丢掉它的 h1 与顶部导航）
    parts = [text]
    for extra in CHAPTER_EXTRAS.get(cid, []):
        p2 = os.path.join(ROOT, folder, extra)
        if not os.path.exists(p2):
            continue
        t2 = io.open(p2, encoding="utf-8").read()
        m2 = re.search(r"^## ", t2, flags=re.M)
        if m2:
            t2 = t2[m2.start():]
        parts.append(t2.rstrip())
    text = "\n\n".join(parts)

    lines = text.split("\n")

    # 标题
    h1 = None
    for k, ln in enumerate(lines):
        if ln.startswith("# "):
            h1 = ln[2:].strip()
            lines = lines[k + 1:]
            break
    title = title or (h1 or cid)

    lines = strip_nav(lines)

    # 按 ## 切小节
    chunks, cur, cur_title = [], [], None
    for ln in lines:
        if ln.startswith("## "):
            if cur_title is not None:
                chunks.append((cur_title, cur))
            elif any(x.strip() for x in cur):
                chunks.append(("引言", cur))
            cur_title, cur = ln[3:].strip(), []
        else:
            cur.append(ln)
    if cur_title is not None:
        chunks.append((cur_title, cur))
    elif any(x.strip() for x in cur):
        chunks.append(("引言", cur))

    sections = []
    n = 0
    for stitle, body in chunks:
        # 去掉首尾空行和分隔线
        while body and (not body[0].strip() or re.match(r"^-{3,}$", body[0].strip())):
            body.pop(0)
        while body and (not body[-1].strip() or re.match(r"^-{3,}$", body[-1].strip())):
            body.pop()
        if not body:
            continue
        n += 1
        sid = "s%d" % n
        slug_map[(cid, slug(stitle))] = sid
        slug_map[(cid, slug(re.sub(r"^\d+[.、]\s*", "", stitle)))] = sid
        html = render_blocks(body, Ctx(slug_map), sid)
        # 任务清单：- [ ] / - [x] → 带方框的 li
        html = re.sub(r"<li>\[ \]\s*", '<li class="task">', html)
        html = re.sub(r"<li>\[[xX]\]\s*", '<li class="task done">', html)
        sections.append({
            "id": sid,
            "title": stitle,
            "short": plain(re.sub(r"^\d+[.、]\s*", "", stitle)),
            "html": html,
        })

    return {
        "id": cid,
        "num": NUMS.get(cid, ""),
        "title": title,
        "heading": h1 or title,
        "subtitle": subtitle,
        "sections": sections,
    }


def main():
    slug_map = {}
    data = [build_chapter(cid, folder, t, sub, slug_map) for cid, folder, t, sub in CHAPTERS]

    # 第二遍：把 h1 里的标题当章节名（同时更新侧边栏）
    total = sum(len(c["sections"]) for c in data)

    tpl = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shell.html"), encoding="utf-8").read()
    out = (tpl
           .replace("/*__DATA__*/[]", json.dumps(data, ensure_ascii=False, indent=1))
           .replace("__TOTAL__", str(total))
           .replace("__BUILD__", "8 章 / %d 节" % total))
    dest = os.path.join(ROOT, "index.html")
    io.open(dest, "w", encoding="utf-8", newline="\n").write(out)
    print("wrote", dest, len(out), "chars,", total, "sections")


if __name__ == "__main__":
    main()
