from statistics import median
from typing import List

from src.models import TextItem, PageRegions
from src.pdf_io import pdf_to_img_coords
from src.text_utils import (
    alpha_ratio,
    clean_candidate_label_text,
    has_alpha,
    is_dimension_like,
    is_generic_metadata,
    is_spec_annotation,
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
    original_text = normalize_text(item.text)
    t, embedded_area_value = clean_candidate_label_text(original_text)
    words = token_count(t)
    alpha_share = alpha_ratio(t)
    punct_share = punctuation_ratio(t)
    char_count = len(t)
    item.embedded_area_value = embedded_area_value

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

    if is_spec_annotation(original_text, t):
        item.text_type = "metadata"
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

    if embedded_area_value is not None:
        score += 4

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


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _area_attachment_score(label: TextItem, area_item: TextItem) -> float | None:
    if label.page_number != area_item.page_number:
        return None

    center_dx = abs(label.cx - area_item.cx)
    center_dy = abs(label.cy - area_item.cy)
    x_overlap = _interval_overlap(label.x0, label.x1, area_item.x0, area_item.x1)
    min_width = max(min(label.width, area_item.width), 1.0)
    x_overlap_ratio = x_overlap / min_width
    stacked_gap = area_item.y0 - label.y1
    max_vertical_gap = max(18.0, label.height * 4.0, area_item.height * 6.0)
    max_horizontal_offset = max(18.0, label.width * 0.75, area_item.width * 0.9)

    if -max(label.height, area_item.height) <= stacked_gap <= max_vertical_gap and center_dx <= max_horizontal_offset:
        return (x_overlap_ratio * 30.0) - (center_dx * 1.5) - (max(stacked_gap, 0.0) * 2.0)

    horizontal_gap = min(abs(area_item.x0 - label.x1), abs(label.x0 - area_item.x1))
    max_horizontal_gap = max(18.0, label.width * 1.1)
    if center_dy <= max(label.height * 1.5, area_item.height * 1.5, 10.0) and horizontal_gap <= max_horizontal_gap:
        return (x_overlap_ratio * 12.0) - (horizontal_gap * 1.8) - center_dy

    return None


def attach_nearby_area_annotations(
    label_candidates: List[TextItem],
    classified_items: List[TextItem],
) -> None:
    area_only_items = [
        item
        for item in classified_items
        if item.embedded_area_value is not None and item.text_type != "room_label"
    ]
    if not area_only_items:
        return

    pair_candidates: List[tuple[float, int, int]] = []
    for label_index, label in enumerate(label_candidates):
        if label.embedded_area_value is not None:
            continue

        for area_index, area_item in enumerate(area_only_items):
            score = _area_attachment_score(label, area_item)
            if score is None or score < -18.0:
                continue
            pair_candidates.append((score, label_index, area_index))

    pair_candidates.sort(reverse=True, key=lambda row: row[0])

    claimed_labels: set[int] = set()
    claimed_areas: set[int] = set()
    for _, label_index, area_index in pair_candidates:
        if label_index in claimed_labels or area_index in claimed_areas:
            continue

        label = label_candidates[label_index]
        area_item = area_only_items[area_index]
        label.embedded_area_value = area_item.embedded_area_value
        claimed_labels.add(label_index)
        claimed_areas.add(area_index)


def prune_final_label_candidates(items: List[TextItem], min_area_sqm: float) -> List[TextItem]:
    out: List[TextItem] = []
    for item in items:
        text = normalize_text(item.text)
        if not text:
            continue

        if item.area_value is not None and item.area_value < min_area_sqm:
            continue

        out.append(item)

    return out
