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
    "COPYRIGHT", "GENERAL", "NOTES", "MUMBAI", "DELHI", "PVT", "LTD",
    "BANK", "STREET", "BRELVI", "GHODA", "FORT", "HOUSE", "GMAIL",
    "COMMERCIAL", "BUILDING", "ROAD", "WEST", "HIGHWAY", "SECTOR"
}

SPACE_HINTS = {
    "LOBBY", "PASSAGE", "STAIRCASE", "PANTRY", "TOILET",
    "LIFT", "A.H.U", "AHU", "OFFICE", "OBSERVATORY"
}

SERVICE_HINTS = {
    "CO-RA", "CO-FA", "FHC", "ELEC", "CHW", "PLUMBING",
    "SHAFT", "PRESSURIZATION", "RWS", "ELV"
}

FRAGMENT_ONLY = {"UP", "DN", "FOR", "EXE", "-", "1", "2", "3", "4", "5"}

ALL_HINTS = SPACE_HINTS | SERVICE_HINTS


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
    t = re.sub(r"\s+", " ", item.text.strip().upper())
    t = t.replace("OBSERVATORY FOR", "FOR OBSERVATORY").strip()
    words = t.split()

    if not t:
        item.text_type = "unknown"
        item.score = 0.0
        return item

    if len(words) == 1 and t in FRAGMENT_ONLY:
        item.text_type = "symbol"
        item.score = 0.0
        return item

    if len(t) > 100 or len(words) > 10:
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

    score = 0.0

    if item.region_type == "drawing":
        score += 20
    elif item.region_type == "border":
        score += 2
    else:
        score -= 15

    hint_hits = sum(1 for h in ALL_HINTS if h in t)
    score += hint_hits * 14

    alpha_count = sum(c.isalpha() for c in t)
    score += min(alpha_count, 12)

    if len(words) <= 8:
        score += 6
    else:
        score -= 8

    if re.search(r"\bLIFT\s+[A-Z]\d+\b", t):
        score += 10
    if re.search(r"\b[FS]\s+LIFT\b", t):
        score += 8
    if re.search(r"\b[FS]\s+LOBBY\b", t):
        score += 8
    if "FOR OBSERVATORY" in t:
        score += 8
    if "PLUMBING SHAFT" in t:
        score += 8
    if "PRESSURIZATION SHAFT" in t:
        score += 8
    if "TOILET/PANTRY" in t:
        score += 8

    if score >= 28:
        item.text_type = "room_label"
    elif score >= 14:
        item.text_type = "unknown"
    else:
        item.text_type = "metadata" if item.region_type == "metadata" else "symbol"

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
    elif space_hits == service_hits and space_hits > 0:
        item.label_category = "space"
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