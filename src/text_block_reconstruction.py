import re
from collections import defaultdict
from typing import List

from src.models import TextItem


def normalize_text(text: str) -> str:
    t = str(text).strip().upper()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("A H U", "A.H.U")
    t = t.replace("F TOILET", "F.TOILET")
    t = t.replace("M TOILET", "M.TOILET")
    t = t.replace("H TOILET", "H.TOILET")
    t = t.replace("STAIR CASE", "STAIRCASE")
    t = t.replace("SHATFT", "SHAFT")
    return t


def _clean_word(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return ""
    if re.fullmatch(r"\d{4,}", t):
        return ""
    return t


def reconstruct_text_blocks(words: List[TextItem]) -> List[TextItem]:
    """
    Reconstruct text using native PDF block_no/line_no grouping.
    This is much more reliable for vector PDFs than custom regrouping.
    """
    if not words:
        return []

    grouped = defaultdict(list)
    for w in words:
        txt = _clean_word(w.text)
        if not txt:
            continue

        nw = TextItem(
            text=txt,
            x0=w.x0,
            y0=w.y0,
            x1=w.x1,
            y1=w.y1,
            source="pdf_block",
            block_no=w.block_no,
            line_no=w.line_no,
            word_no=w.word_no,
        )
        grouped[(w.block_no, w.line_no)].append(nw)

    blocks = []
    for (block_no, line_no), items in grouped.items():
        items = sorted(items, key=lambda x: x.word_no)

        text = " ".join(i.text for i in items).strip()
        text = re.sub(r"\s+", " ", text)

        if not text:
            continue

        x0 = min(i.x0 for i in items)
        y0 = min(i.y0 for i in items)
        x1 = max(i.x1 for i in items)
        y1 = max(i.y1 for i in items)

        blocks.append(
            TextItem(
                text=text,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                source="pdf_block",
                block_no=block_no,
                line_no=line_no,
                word_no=0,
            )
        )

    return sorted(blocks, key=lambda b: (b.y0, b.x0))


def deduplicate_blocks(items: List[TextItem]) -> List[TextItem]:
    seen = set()
    out = []
    for item in items:
        key = (item.text, round(item.cx, 1), round(item.cy, 1))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out