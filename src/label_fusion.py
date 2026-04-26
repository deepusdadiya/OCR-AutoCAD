import re
from typing import List, Tuple

from src.models import TextItem
from src.text_utils import normalize_text, similarity_ratio


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _size_compatible(a: TextItem, b: TextItem) -> bool:
    if a.font_size <= 0 or b.font_size <= 0:
        return True
    return similarity_ratio(a.font_size, b.font_size) >= 0.8


def _horizontal_pair_score(a: TextItem, b: TextItem) -> float:
    x_overlap = _interval_overlap(a.x0, a.x1, b.x0, b.x1)
    min_width = max(min(a.width, b.width), 1.0)
    x_overlap_ratio = x_overlap / min_width

    y_gap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    max_gap = max(12.0, max(a.height, b.height) * 1.5)

    if x_overlap_ratio < 0.35 or y_gap > max_gap:
        return -1.0

    score = x_overlap_ratio * 10.0 - (y_gap / max_gap) * 4.0
    if a.block_no == b.block_no:
        score += 3.0
    return score


def _vertical_pair_score(a: TextItem, b: TextItem) -> float:
    y_overlap = _interval_overlap(a.y0, a.y1, b.y0, b.y1)
    min_height = max(min(a.height, b.height), 1.0)
    y_overlap_ratio = y_overlap / min_height

    x_gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    max_gap = max(10.0, max(a.width, b.width) * 1.6)

    if y_overlap_ratio < 0.45 or x_gap > max_gap:
        return -1.0

    score = y_overlap_ratio * 10.0 - (x_gap / max_gap) * 4.0
    if a.block_no == b.block_no:
        score += 3.0
    return score


def _pair_score(a: TextItem, b: TextItem) -> float:
    if a.page_number != b.page_number or a.region_type != b.region_type:
        return -1.0

    if a.orientation != b.orientation:
        return -1.0

    if not _size_compatible(a, b):
        return -1.0

    if a.orientation == "horizontal":
        return _horizontal_pair_score(a, b)

    return _vertical_pair_score(a, b)


def _ordered_group(items: List[TextItem]) -> List[TextItem]:
    if not items:
        return []

    orientation = items[0].orientation
    same_block = len({item.block_no for item in items}) == 1

    if orientation == "horizontal":
        if same_block:
            return sorted(items, key=lambda item: (item.line_no, item.x0))
        return sorted(items, key=lambda item: (item.y0, item.x0))

    if same_block:
        return sorted(items, key=lambda item: (item.line_no, item.x0))
    return sorted(items, key=lambda item: (-item.x0, item.y0))


def _merge_text(parts: List[TextItem]) -> str:
    tokens = []
    for part in parts:
        for token in normalize_text(part.text).split():
            if not tokens or tokens[-1] != token:
                tokens.append(token)
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def _merge_pair(parts: List[TextItem]) -> TextItem:
    ordered = _ordered_group(parts)
    base = ordered[0]

    return TextItem(
        text=_merge_text(ordered),
        x0=min(item.x0 for item in ordered),
        y0=min(item.y0 for item in ordered),
        x1=max(item.x1 for item in ordered),
        y1=max(item.y1 for item in ordered),
        source="fused_label",
        region_type=base.region_type,
        text_type=base.text_type,
        label_category="label",
        score=max(item.score for item in ordered),
        block_no=base.block_no,
        line_no=base.line_no,
        word_no=0,
        page_number=base.page_number,
        font_size=max(item.font_size for item in ordered),
        dir_x=base.dir_x,
        dir_y=base.dir_y,
    )


def fuse_label_candidates(label_candidates: List[TextItem]) -> List[TextItem]:
    if not label_candidates:
        return []

    items = sorted(label_candidates, key=lambda item: (item.page_number, item.y0, item.x0))
    pair_candidates: List[Tuple[float, int, int]] = []

    for i, item in enumerate(items):
        for j in range(i + 1, len(items)):
            other = items[j]
            score = _pair_score(item, other)
            if score >= 4.0:
                pair_candidates.append((score, i, j))

    pair_candidates.sort(reverse=True, key=lambda row: row[0])

    used = set()
    fused: List[TextItem] = []
    for _, i, j in pair_candidates:
        if i in used or j in used:
            continue

        merged = _merge_pair([items[i], items[j]])
        fused.append(merged)
        used.add(i)
        used.add(j)

    for idx, item in enumerate(items):
        if idx not in used:
            item.text = normalize_text(item.text)
            fused.append(item)

    out = []
    seen = set()
    for item in fused:
        key = (normalize_text(item.text), round(item.cx, 1), round(item.cy, 1))
        if key in seen:
            continue
        seen.add(key)
        item.text = normalize_text(item.text)
        out.append(item)

    return sorted(out, key=lambda item: (item.page_number, item.y0, item.x0))