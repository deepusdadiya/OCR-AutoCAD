import re
from typing import List

from src.models import TextItem, PageRegions
from src.pdf_io import pdf_to_img_coords


DIMENSION_PATTERNS = [
    r"^\d+(\.\d+)?$",
    r"^\d{3,}$",
    r"^\d+(\.\d+)?\s*(MM|CM|M|SQM|SQMM)$",
]

METADATA_KEYWORDS = {
    "ARCHITECT", "CONSULTANT", "ENGINEERING", "PROJECT", "EMAIL", "ADDRESS",
    "DRAWING", "DOCUMENT", "REVISION", "DATE", "SCALE", "DRAWN", "APPROVED",
    "COPYRIGHT", "GENERAL", "NOTES", "MUMBAI", "DELHI", "PVT", "LTD"
}

ROOM_HINTS = {
    "LIFT", "LOBBY", "PASSAGE", "AHU", "A.H.U", "STAIRCASE", "PANTRY",
    "TOILET", "SHAFT", "ELEC", "PLUMBING", "FHC", "CHW", "CO-FA", "CO-RA"
}

REJECT_TOKENS = {
    "F", "FOR", "B1", "B2", "B3", "B4", "B5", "EXE", "UP", "DN", "RWS", "ELV"
}

SPACE_HINTS = {
    "LOBBY", "PASSAGE", "STAIRCASE", "PANTRY",
    "TOILET", "LIFT", "A.H.U", "AHU", "OFFICE"
}

SERVICE_HINTS = {
    "CO-RA", "CO-FA", "FHC", "ELEC", "CHW",
    "PLUMBING", "SHAFT", "PRESSURIZATION"
}


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


def _looks_like_dimension(text: str) -> bool:
    t = text.strip().upper()
    return any(re.fullmatch(p, t) for p in DIMENSION_PATTERNS)


def classify_text_item(item: TextItem) -> TextItem:
    t = item.text.strip().upper()
    words = t.split()

    if not t:
        item.text_type = "unknown"
        item.score = 0.0
        return item

    if len(t) > 60 or len(words) > 5:
        item.text_type = "metadata"
        item.score = 0.0
        return item

    if _looks_like_dimension(t):
        item.text_type = "dimension"
        item.score = 0.0
        return item

    if any(k in t for k in METADATA_KEYWORDS):
        item.text_type = "metadata"
        item.score = 0.0
        return item

    if len(words) == 1 and t in REJECT_TOKENS:
        item.text_type = "symbol"
        item.score = 0.0
        return item

    score = 0.0

    if item.region_type == "drawing":
        score += 20
    elif item.region_type == "border":
        score += 4
    else:
        score -= 12

    matched_hints = sum(1 for h in ROOM_HINTS if h in t)
    score += matched_hints * 18

    alpha_count = sum(c.isalpha() for c in t)
    score += min(alpha_count, 10)

    if len(words) == 1:
        if matched_hints > 0:
            score += 4
        else:
            score -= 8
    elif len(words) <= 3:
        score += 6
    else:
        score -= 10

    if re.search(r"\d{4,}", t):
        score -= 20

    if all(w in REJECT_TOKENS for w in words):
        score -= 20

    if score >= 32:
        item.text_type = "room_label"
    elif score >= 16:
        item.text_type = "unknown"
    elif item.region_type == "metadata":
        item.text_type = "metadata"
    else:
        item.text_type = "symbol"

    item.score = score
    return item


def assign_label_category(item: TextItem) -> TextItem:
    t = item.text.upper()

    space_hits = sum(1 for h in SPACE_HINTS if h in t)
    service_hits = sum(1 for h in SERVICE_HINTS if h in t)

    if space_hits > service_hits and space_hits > 0:
        item.label_category = "space"
    elif service_hits > space_hits and service_hits > 0:
        item.label_category = "service"
    else:
        item.label_category = "unknown"

    return item


def classify_text_items(items: List[TextItem]) -> List[TextItem]:
    out = []
    for item in items:
        item = classify_text_item(item)
        item = assign_label_category(item)
        out.append(item)
    return out


def keep_room_label_candidates(items: List[TextItem]) -> List[TextItem]:
    return [i for i in items if i.text_type == "room_label"]