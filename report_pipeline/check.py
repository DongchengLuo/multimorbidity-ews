"""Information-completeness check: every number, sentence and table cell in the md must appear in the HTML."""
from __future__ import annotations

import re


def strip_html(s):
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.S)
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    return s


def norm(s):
    return re.sub(r"\s+", "", s)


def demd(s):
    """Strip markdown markup, keep the body text only."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"(?m)^\s*[-*+]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+[.、)]\s+", "", s)
    s = re.sub(r"(?m)^\s*>\s?", "", s)
    s = re.sub(r"(?m)^#+\s*", "", s)
    return s


def check(md, html, verbose=True):
    text = norm(strip_html(html))

    md = demd(md)
    nums = re.findall(r"\d+(?:\.\d+)?", md)
    miss_nums = [n for n in dict.fromkeys(nums) if n not in text]

    body = re.sub(r"^#.*$", "", md, flags=re.M)
    body = re.sub(r"^\|.*$", "", body, flags=re.M)
    sents = [s for s in re.split(r"[。！？\n]", body) if len(norm(s)) >= 12]
    miss_sents = []
    for s in sents:
        key = norm(s)[:40]
        if key and key not in text:
            miss_sents.append(s.strip()[:60])

    cells = []
    for line in md.split("\n"):
        if line.strip().startswith("|") and not re.match(r"^\|[\s\-:|]+\|$", line.strip()):
            for c in line.strip().strip("|").split("|"):
                c = norm(c)
                if len(c) >= 4:
                    cells.append(c)
    miss_cells = [c for c in dict.fromkeys(cells) if c not in text]

    if verbose:
        print(f"Numbers: {len(set(nums))}, missing {len(miss_nums)}")
        if miss_nums:
            print("  ", miss_nums[:30])
        print(f"Sentences: {len(sents)}, missing {len(miss_sents)}")
        for s in miss_sents[:15]:
            print("   -", s)
        print(f"Table cells: {len(set(cells))}, missing {len(miss_cells)}")
        for c in miss_cells[:15]:
            print("   -", c)
    return not (miss_nums or miss_sents or miss_cells)
