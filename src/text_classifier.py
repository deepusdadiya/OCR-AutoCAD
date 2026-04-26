from statistics import median
from typing import List

from src.models import TextItem, PageRegions
from src.pdf_io import pdf_to_img_coords
from src.text_utils import (
    alpha_ratio,
    has_alpha,
    is_dimension_like,
    is_generic_metadata,
    is_short_fragment,
    normalize_text,
    punctuation_ratio,
    token_count,
)


def _in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    x, y, w, h = bbox
    px, py = point
    return x <= px <= x + w and y <= py <= y + h


def assign_region_type(
    items: List[TextItem],
    page_regions: PageRegions,
    page_w: float,
    page_h: float,
    img_w: int,
    img_h: int,
) -> List[TextItem]:
    for item in items:
        p = pdf_to_img_coords(item.cx, item.cy, page_w, page_h, img_w, img_h)

        if _in_bbox(p, page_regions.drawing_bbox_img):
            item.region_type = "drawing"
        elif any(_in_bbox(p, b) for b in page_regions.metadata_bboxes_img):
            item.region_type = "metadata"
        else:
            item.region_type = "border"

    return items


def _classify_label_shape(text: str) -> str:
    words = token_count(text)

    if words == 1 and len(text) <= 6:
        return "short_label"

    if any(ch.isdigit() for ch in text) or any(ch in text for ch in "/-."):
        return "qualified_label"

    return "label"


def classify_text_item(item: TextItem, drawing_font_median: float) -> TextItem:
    t = normalize_text(item.text)
    words = token_count(t)
    alpha_share = alpha_ratio(t)
    punct_share = punctuation_ratio(t)
    char_count = len(t)

    if not t:
        item.text_type = "unknown"
        item.score = 0.0
        item.label_category = "unknown"
        return item

    if is_short_fragment(t):
        item.text_type = "symbol"
        item.score = 0.0
        item.label_category = "unknown"
        return item

    if not has_alpha(t):
        item.text_type = "dimension" if is_dimension_like(t) else "symbol"
        item.score = 0.0
        item.label_category = "unknown"
        return item

    if is_dimension_like(t):
        item.text_type = "dimension"
        item.score = 0.0
        item.label_category = "unknown"
        return item

    if is_generic_metadata(t):
        item.text_type = "metadata"
        item.score = 0.0
        item.label_category = "unknown"
        return item

    score = 0.0

    if item.region_type == "drawing":
        score += 28
    elif item.region_type == "border":
        score -= 20
    else:
        score -= 35

    if 1 <= words <= 4:
        score += 10
    elif words <= 6:
        score += 5
    else:
        score -= 12

    if 2 <= char_count <= 30:
        score += 8
    elif char_count <= 40:
        score += 3
    else:
        score -= 10

    if alpha_share >= 0.55:
        score += 8
    elif alpha_share >= 0.35:
        score += 3
    else:
        score -= 8

    if punct_share > 0.35:
        score -= 8
    elif punct_share > 0.2:
        score -= 3

    if drawing_font_median > 0 and item.font_size > 0:
        ratio = min(item.font_size, drawing_font_median) / max(item.font_size, drawing_font_median)
        if ratio >= 0.8:
            score += 6
        elif ratio >= 0.65:
            score += 2
        else:
            score -= 4

    if item.orientation == "vertical":
        score += 2

    item.score = round(score, 2)

    if score >= 26:
        item.text_type = "room_label"
    elif score >= 14:
        item.text_type = "unknown"
    else:
        item.text_type = "metadata" if item.region_type != "drawing" else "symbol"

    item.label_category = _classify_label_shape(t) if item.text_type == "room_label" else "unknown"
    item.text = t
    return item


def classify_text_items(items: List[TextItem]) -> List[TextItem]:
    drawing_font_sizes = [
        item.font_size
        for item in items
        if item.region_type == "drawing" and item.font_size > 0 and has_alpha(item.text)
    ]
    drawing_font_median = median(drawing_font_sizes) if drawing_font_sizes else 0.0

    out = []
    for item in items:
        out.append(classify_text_item(item, drawing_font_median))
    return out


def keep_room_label_candidates(items: List[TextItem]) -> List[TextItem]:
    return [item for item in items if item.text_type == "room_label" and item.region_type == "drawing"]