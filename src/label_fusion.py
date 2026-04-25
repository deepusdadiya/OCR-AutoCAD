import re
from typing import List, Dict, Any
from src.models import TextItem


ANCHOR_WORDS = {
    "LIFT", "LOBBY", "PASSAGE", "STAIRCASE", "PANTRY",
    "TOILET", "A.H.U", "AHU", "OFFICE",
    "SHAFT", "PLUMBING", "PRESSURIZATION",
    "CO-RA", "CO-FA", "FHC", "ELEC", "CHW",
    "RWS", "ELV"
}


def _norm(text: str) -> str:
    t = str(text).strip().upper()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("A H U", "A.H.U")
    t = t.replace("F TOILET", "F.TOILET")
    t = t.replace("M TOILET", "M.TOILET")
    t = t.replace("H TOILET", "H.TOILET")
    t = t.replace("OBSERVATORY FOR", "FOR OBSERVATORY")
    t = t.replace("SHATFT", "SHAFT")
    return t


def _is_anchor(text: str) -> bool:
    t = _norm(text)
    return t in ANCHOR_WORDS or any(t.startswith(a + " ") for a in ANCHOR_WORDS)


def _is_code_like(text: str) -> bool:
    t = _norm(text)
    return bool(re.fullmatch(r"[A-Z]\d+", t))


def _is_number_like(text: str) -> bool:
    t = _norm(text)
    return bool(re.fullmatch(r"\d+", t))


def _is_prefix_like(text: str) -> bool:
    t = _norm(text)
    return t in {"F", "S"}


def _is_phrase_tail_like(text: str) -> bool:
    t = _norm(text)
    return t in {"SHAFT", "OBSERVATORY", "RWS", "ELV"} or t.startswith("FOR")


def _distance(a: TextItem, b: TextItem) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def fuse_label_candidates(label_candidates: List[TextItem]) -> List[TextItem]:
    """
    Generic fusion:
    - attach prefix tokens like F / S to nearest anchor
    - attach code tokens like B1 to nearest anchor
    - attach short phrase tails like FOR / OBSERVATORY / SHAFT
    """
    if not label_candidates:
        return []

    items = sorted(label_candidates, key=lambda x: (x.cy, x.cx))
    used = set()
    fused: List[TextItem] = []

    for i, item in enumerate(items):
        if i in used:
            continue

        base = _norm(item.text)

        if not _is_anchor(base):
            continue

        group = [item]
        used.add(i)

        # dynamic local neighborhood
        x_tol = max(45.0, item.height * 8.0)
        y_tol = max(16.0, item.height * 3.0)

        # prefix candidate: nearest left short token
        prefix_idx = None
        prefix_gap = None

        # right-side candidates
        right_neighbors = []

        for j, other in enumerate(items):
            if j == i or j in used:
                continue

            ot = _norm(other.text)
            dx = other.cx - item.cx
            dy = abs(other.cy - item.cy)

            if dy > y_tol:
                continue

            # left-side prefix
            if other.x1 <= item.x0:
                gap = item.x0 - other.x1
                if gap <= x_tol and _is_prefix_like(ot):
                    if prefix_gap is None or gap < prefix_gap:
                        prefix_idx = j
                        prefix_gap = gap

            # right-side suffix / qualifier
            if other.x0 >= item.x1:
                gap = other.x0 - item.x1
                if gap <= x_tol and (
                    _is_code_like(ot) or _is_number_like(ot) or _is_phrase_tail_like(ot)
                ):
                    right_neighbors.append((gap, j, other))

        if prefix_idx is not None:
            group.append(items[prefix_idx])
            used.add(prefix_idx)

        right_neighbors = sorted(right_neighbors, key=lambda x: x[0])

        # attach only closest 2 right tokens max
        for _, j, other in right_neighbors[:2]:
            group.append(other)
            used.add(j)

        group = sorted(group, key=lambda g: g.x0)

        text = " ".join(_norm(g.text) for g in group)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("OBSERVATORY FOR", "FOR OBSERVATORY")

        fused.append(
            TextItem(
                text=text,
                x0=min(g.x0 for g in group),
                y0=min(g.y0 for g in group),
                x1=max(g.x1 for g in group),
                y1=max(g.y1 for g in group),
                source="fused_label",
                region_type=item.region_type,
                text_type=item.text_type,
                label_category=item.label_category,
                score=max(g.score for g in group),
                block_no=item.block_no,
                line_no=item.line_no,
                word_no=item.word_no,
            )
        )

    # keep untouched non-fragment labels that were not fused into anchors
    for i, item in enumerate(items):
        if i in used:
            continue

        t = _norm(item.text)

        # drop isolated fragments
        if t in {"UP", "DN", "FOR", "EXE", "-", "1", "2", "3", "4", "5"}:
            continue

        fused.append(item)

    # deduplicate by text+position
    out = []
    seen = set()
    for item in fused:
        key = (_norm(item.text), round(item.cx, 1), round(item.cy, 1))
        if key not in seen:
            seen.add(key)
            item.text = _norm(item.text)
            out.append(item)

    return sorted(out, key=lambda x: (x.cy, x.cx))