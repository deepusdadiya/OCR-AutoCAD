import re
from typing import List

from src.models import TextItem


STOP_TOKENS = {
    "UP", "DN", "FOR", "F", "S", "N", "E", "W",
    "B1", "B2", "B3", "B4", "B5", "EXE", "RWS", "ELV", "DCAB"
}


def normalize_text(text: str) -> str:
    t = str(text).strip().upper()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("A H U", "A.H.U")
    t = t.replace("F TOILET", "F.TOILET")
    t = t.replace("M TOILET", "M.TOILET")
    t = t.replace("H TOILET", "H.TOILET")
    t = t.replace("STAIR CASE", "STAIRCASE")
    return t


def _clean_phrase_tokens(tokens: List[str]) -> List[str]:
    out = []
    for tok in tokens:
        tok = normalize_text(tok)
        if not tok:
            continue
        if tok in STOP_TOKENS:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            continue
        if re.fullmatch(r"\d{3,}", tok):
            continue
        out.append(tok)
    return out


def group_words_into_phrases(
    words: List[TextItem],
    y_tol: float = 6.0,
    x_gap_tol: float = 22.0,
) -> List[TextItem]:
    if not words:
        return []

    words = sorted(words, key=lambda w: (round(w.y0, 1), w.x0))
    used = [False] * len(words)
    merged = []

    for i, w in enumerate(words):
        if used[i]:
            continue

        cluster = [w]
        used[i] = True
        last = w

        for j in range(i + 1, len(words)):
            if used[j]:
                continue

            nxt = words[j]
            same_line = abs(nxt.cy - w.cy) <= y_tol
            gap = nxt.x0 - last.x1
            close_x = -2 <= gap <= x_gap_tol

            if same_line and close_x:
                cluster.append(nxt)
                used[j] = True
                last = nxt
            elif nxt.cy - w.cy > y_tol:
                break

        tokens = _clean_phrase_tokens([c.text for c in cluster])
        text = " ".join(tokens).strip()
        text = re.sub(r"\s+", " ", text)

        if not text:
            continue

        x0 = min(c.x0 for c in cluster)
        y0 = min(c.y0 for c in cluster)
        x1 = max(c.x1 for c in cluster)
        y1 = max(c.y1 for c in cluster)

        merged.append(TextItem(text=text, x0=x0, y0=y0, x1=x1, y1=y1, source="pdf_phrase"))

    return merged


def deduplicate_phrases(items: List[TextItem]) -> List[TextItem]:
    seen = set()
    out = []
    for item in items:
        key = (item.text, round(item.cx, 0), round(item.cy, 0))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out