import re
from typing import List

from src.models import TextItem


ROOM_KEYWORDS = [
    "LIFT", "LOBBY", "PASSAGE", "A.H.U", "AHU",
    "STAIRCASE", "PANTRY", "TOILET", "SHAFT",
    "PLUMBING", "ELEC", "CHW", "CO-FA", "CO-RA",
    "OBSERVATORY", "FHC"
]

BAD_FULL_PATTERNS = [
    r"(CONSULTANT|ARCHITECT|ENGINEERING|PROJECT|EMAIL|ADDRESS)",
    r"(MUMBAI|GURUGRAM|DELHI|PVT|LTD)",
    r"(COPYRIGHT|GENERAL NOTES|COMMENCEMENT OF WORK)",
    r"(DOCUMENT NUMBER|PURPOSE OF ISSUE|DRAWING TITLE)",
]

BAD_TOKENS = {
    "UP", "DN", "RWS", "ELV", "DCAB", "REV", "DATE", "DRAWN", "SCALE"
}


def normalize_text(text: str) -> str:
    t = str(text).strip().upper()
    t = re.sub(r"\s+", " ", t)

    replacements = {
        "A H U": "A.H.U",
        "F TOILET": "F.TOILET",
        "M TOILET": "M.TOILET",
        "H TOILET": "H.TOILET",
        "LIFT - 1": "LIFT-1",
        "LIFT - 2": "LIFT-2",
        "STAIR CASE": "STAIRCASE",
    }

    for old, new in replacements.items():
        t = t.replace(old, new)

    return t


def looks_like_dimension(text: str) -> bool:
    t = normalize_text(text)
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return True
    if re.fullmatch(r"\d{3,}", t):
        return True
    if re.fullmatch(r"\d+(\.\d+)?\s*(MM|CM|M|SQM|SQMM)", t):
        return True
    return False


def looks_like_noise(text: str) -> bool:
    t = normalize_text(text)

    for pattern in BAD_FULL_PATTERNS:
        if re.search(pattern, t):
            return True

    if len(t) <= 1:
        return True

    return False


def clean_token(token: str) -> str:
    t = normalize_text(token)
    if looks_like_dimension(t):
        return ""
    if t in BAD_TOKENS:
        return ""
    if re.fullmatch(r"[0-9]+", t):
        return ""
    return t


def looks_like_room_label(text: str) -> bool:
    t = normalize_text(text)

    if not t or len(t) < 2 or len(t) > 35:
        return False
    if looks_like_dimension(t) or looks_like_noise(t):
        return False

    return any(k in t for k in ROOM_KEYWORDS)


def merge_words_safely(words: List[TextItem], y_tol: float = 4.0, x_gap_tol: float = 18.0) -> List[TextItem]:
    """
    Merge only nearby words on the same line.
    Much stricter than the previous version.
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: (round(w.y0, 1), w.x0))
    merged = []
    used = [False] * len(words)

    for i, w in enumerate(words):
        if used[i]:
            continue

        current_words = [w]
        used[i] = True
        last = w

        for j in range(i + 1, len(words)):
            if used[j]:
                continue

            nxt = words[j]

            same_line = abs(nxt.y0 - last.y0) <= y_tol
            close_x = (nxt.x0 - last.x1) <= x_gap_tol

            if same_line and close_x:
                current_words.append(nxt)
                used[j] = True
                last = nxt
            elif nxt.y0 - w.y0 > y_tol:
                break

        text = " ".join(clean_token(x.text) for x in current_words).strip()
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        x0 = min(x.x0 for x in current_words)
        y0 = min(x.y0 for x in current_words)
        x1 = max(x.x1 for x in current_words)
        y1 = max(x.y1 for x in current_words)

        merged.append(TextItem(text=text, x0=x0, y0=y0, x1=x1, y1=y1, source="pdf_line"))

    return merged


def filter_room_like_texts(words: List[TextItem]) -> List[TextItem]:
    out = []

    for item in words:
        t = normalize_text(item.text)

        # split very suspicious long phrases and also keep original if valid
        if len(t.split()) > 4:
            for token in t.split():
                token = clean_token(token)
                if token and looks_like_room_label(token):
                    out.append(TextItem(text=token, x0=item.x0, y0=item.y0, x1=item.x1, y1=item.y1, source=item.source))
            continue

        t = normalize_text(" ".join(clean_token(tok) for tok in t.split()))
        if not t:
            continue

        if looks_like_room_label(t):
            item.text = t
            out.append(item)

    return out


def deduplicate_texts(items: List[TextItem]) -> List[TextItem]:
    seen = set()
    out = []

    for item in items:
        key = (item.text, round(item.cx, 0), round(item.cy, 0))
        if key not in seen:
            seen.add(key)
            out.append(item)

    return out


def score_label(text: str) -> float:
    t = normalize_text(text)
    score = 0.0

    for kw in ROOM_KEYWORDS:
        if kw in t:
            score += 10

    score += min(sum(c.isalpha() for c in t), 12)

    # penalties
    if any(tok in t.split() for tok in BAD_TOKENS):
        score -= 12
    if re.search(r"\d{3,}", t):
        score -= 15
    if len(t.split()) > 4:
        score -= 10

    return score


def best_label_from_texts(items: List[TextItem]) -> tuple[str, float]:
    if not items:
        return "UNNAMED_SPACE", 0.0

    scored = [(item.text, score_label(item.text)) for item in items]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    best_text, best_score = scored[0]

    if best_score <= 0:
        return "UNNAMED_SPACE", 0.0

    confidence = min(best_score / 35.0, 0.99)
    return best_text, confidence