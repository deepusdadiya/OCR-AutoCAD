import re
from typing import List

from src.models import TextItem


ANCHOR_WORDS = {
    "LIFT", "LOBBY", "SHAFT", "PANTRY", "PASSAGE", "STAIRCASE",
    "TOILET", "A.H.U", "AHU", "OFFICE", "PLUMBING",
    "PRESSURIZATION", "FHC", "CO-RA", "CO-FA", "ELEC", "CHW"
}


def _normalize(text: str) -> str:
    t = str(text).strip().upper()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("A H U", "A.H.U")
    t = t.replace("F TOILET", "F.TOILET")
    t = t.replace("M TOILET", "M.TOILET")
    t = t.replace("H TOILET", "H.TOILET")
    t = t.replace("SHATFT", "SHAFT")
    return t


def _is_short_qualifier(token: str) -> bool:
    t = _normalize(token)

    if re.fullmatch(r"[A-Z]\d+", t):   # B1, C2, etc.
        return True
    if re.fullmatch(r"\d+", t):        # 1, 2
        return True
    if t in {"F", "S", "FOR", "OBSERVATORY", "RWS", "ELV", "-", "UP", "DN"}:
        return True
    return False


def _is_anchor(text: str) -> bool:
    t = _normalize(text)
    return any(a in t for a in ANCHOR_WORDS)


def _distance(a: TextItem, b: TextItem) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def expand_label_blocks(blocks: List[TextItem]) -> List[TextItem]:
    """
    Generic label expansion:
    - start from anchor-like blocks
    - absorb nearby short qualifier fragments
    - reconstruct richer labels
    """
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda x: (x.cy, x.cx))
    used_neighbors = set()
    expanded = []

    for i, block in enumerate(blocks):
        text = _normalize(block.text)

        if not _is_anchor(text):
            expanded.append(block)
            continue

        group = [block]

        # dynamic neighborhood based on text height
        x_radius = max(25.0, block.height * 6.0)
        y_radius = max(12.0, block.height * 2.5)

        for j, other in enumerate(blocks):
            if i == j:
                continue

            ot = _normalize(other.text)

            dx = abs(other.cx - block.cx)
            dy = abs(other.cy - block.cy)

            if dx <= x_radius and dy <= y_radius:
                if _is_short_qualifier(ot):
                    group.append(other)
                    used_neighbors.add(j)

        # sort left-to-right, then slightly by y
        group = sorted(group, key=lambda g: (round(g.cy, 1), g.x0))

        merged_text = " ".join(_normalize(g.text) for g in group)
        merged_text = re.sub(r"\s+", " ", merged_text).strip()

        # small cleanup for ordering / noise
        merged_text = merged_text.replace("OBSERVATORY FOR", "FOR OBSERVATORY")
        merged_text = merged_text.replace("UP", "").replace("DN", "")
        merged_text = re.sub(r"\s+", " ", merged_text).strip()

        x0 = min(g.x0 for g in group)
        y0 = min(g.y0 for g in group)
        x1 = max(g.x1 for g in group)
        y1 = max(g.y1 for g in group)

        expanded.append(
            TextItem(
                text=merged_text,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                source="expanded_block",
            )
        )

    # also keep non-anchor neighbors that were not absorbed anywhere
    for j, block in enumerate(blocks):
        if j in used_neighbors:
            continue
        if not _is_anchor(block.text):
            expanded.append(block)

    # deduplicate
    seen = set()
    out = []
    for item in expanded:
        key = (_normalize(item.text), round(item.cx, 1), round(item.cy, 1))
        if key not in seen:
            seen.add(key)
            item.text = _normalize(item.text)
            out.append(item)

    return out