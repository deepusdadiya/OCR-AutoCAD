import re
from collections import defaultdict
from typing import Dict, Any, List

from src.models import TextItem


def _norm(text: str) -> str:
    t = str(text).strip().upper()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("A H U", "A.H.U")
    t = t.replace("F TOILET", "F.TOILET")
    t = t.replace("M TOILET", "M.TOILET")
    t = t.replace("H TOILET", "H.TOILET")
    t = t.replace("STAIR CASE", "STAIRCASE")
    t = t.replace("SHATFT", "SHAFT")
    t = t.replace(" - ", "-")
    return t


def _word_tokens(text: str) -> List[str]:
    return [tok for tok in re.split(r"\s+", _norm(text)) if tok]


def _looks_like_qualifier(tok: str) -> bool:
    if re.fullmatch(r"[A-Z]\d+", tok):   # B1, C2
        return True
    if re.fullmatch(r"\d+", tok):        # 1, 2
        return True
    if tok in {"F", "S", "FOR", "OBSERVATORY", "SHAFT", "RWS", "ELV"}:
        return True
    return False


def _reading_order(texts: List[TextItem]) -> List[TextItem]:
    return sorted(texts, key=lambda t: (round(t.cy, 1), t.x0))


def _combine_local_texts(texts: List[TextItem]) -> str:
    """
    Reconstruct a region-local label from only the texts assigned to that region.
    This is generic because it uses local reading order and qualifiers rather than hardcoded names.
    """
    if not texts:
        return "UNNAMED_SPACE"

    texts = _reading_order(texts)

    # Prefer stronger text items first
    texts_sorted = sorted(texts, key=lambda t: (t.score, -t.width), reverse=True)

    anchor = texts_sorted[0]
    anchor_text = _norm(anchor.text)
    pieces = [anchor_text]

    # Attach only near local neighbors
    local_neighbors = []
    for t in texts:
        if t is anchor:
            continue

        dx = abs(t.cx - anchor.cx)
        dy = abs(t.cy - anchor.cy)

        if dx <= max(anchor.height * 10, 60) and dy <= max(anchor.height * 2.5, 18):
            local_neighbors.append(t)

    local_neighbors = _reading_order(local_neighbors)

    for n in local_neighbors:
        nt = _norm(n.text)

        # ignore obvious metadata garbage
        if nt in {"UP", "DN", "EXE"}:
            continue

        # append small qualifiers or meaningful nearby tokens
        if _looks_like_qualifier(nt) or len(nt.split()) <= 3:
            pieces.append(nt)

    candidate = " ".join(pieces)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    # cleanup ordering issues
    candidate = candidate.replace("OBSERVATORY FOR", "FOR OBSERVATORY")

    # remove duplicate repeated immediate tokens
    toks = candidate.split()
    cleaned = []
    for tok in toks:
        if not cleaned or cleaned[-1] != tok:
            cleaned.append(tok)
    candidate = " ".join(cleaned)

    return candidate if candidate else "UNNAMED_SPACE"


def resolve_region_labels(region_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for payload in region_payloads:
        region = payload["region"]
        texts = payload["texts"]

        label = _combine_local_texts(texts)

        out.append(
            {
                "region": region,
                "texts": texts,
                "resolved_label": label,
            }
        )

    return out


def build_instance_names(resolved_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Number repeated resolved labels as #1, #2, ...
    Keep already-qualified names like LIFT B1 unchanged.
    """
    grouped = defaultdict(list)

    for payload in resolved_payloads:
        base = _norm(payload["resolved_label"])
        grouped[base].append(payload)

    final = []
    for base_name, items in grouped.items():
        items = sorted(items, key=lambda p: (p["region"].centroid[1], p["region"].centroid[0]))

        already_specific = bool(re.search(r"\b[A-Z]\d+\b", base_name))
        if len(items) == 1 or already_specific:
            for p in items:
                p["instance_name"] = base_name
                final.append(p)
        else:
            for i, p in enumerate(items, start=1):
                p["instance_name"] = f"{base_name} #{i}"
                final.append(p)

    return final