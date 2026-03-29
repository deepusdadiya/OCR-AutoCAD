from typing import Tuple, List
import cv2
import numpy as np
from PIL import Image

from src.models import PageRegions


def _merge_boxes(boxes: List[Tuple[int, int, int, int]], pad: int = 10) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    merged = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        x, y, w, h = box
        expanded = (x - pad, y - pad, w + 2 * pad, h + 2 * pad)

        placed = False
        for i, mb in enumerate(merged):
            mx, my, mw, mh = mb
            if not (expanded[0] > mx + mw or mx > expanded[0] + expanded[2] or
                    expanded[1] > my + mh or my > expanded[1] + expanded[3]):
                nx0 = min(mx, expanded[0])
                ny0 = min(my, expanded[1])
                nx1 = max(mx + mw, expanded[0] + expanded[2])
                ny1 = max(my + mh, expanded[1] + expanded[3])
                merged[i] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
                placed = True
                break
        if not placed:
            merged.append(expanded)

    return merged


def detect_page_regions(
    image: Image.Image,
    min_component_area: int = 8000,
    drawing_region_padding: int = 20,
) -> PageRegions:
    """
    Generic detection of:
    - main drawing region
    - metadata/text blocks outside the main drawing region
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Dark content = white foreground
    _, bw = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_component_area:
            components.append((x, y, w, h, area))

    if not components:
        h, w = gray.shape
        return PageRegions(drawing_bbox_img=(0, 0, w, h), metadata_bboxes_img=[])

    # Assume the drawing region is the largest central component
    h, w = gray.shape
    page_cx, page_cy = w / 2, h / 2

    scored = []
    for x, y, bw_, bh_, area in components:
        cx = x + bw_ / 2
        cy = y + bh_ / 2
        center_penalty = ((cx - page_cx) ** 2 + (cy - page_cy) ** 2) ** 0.5
        score = area - 0.15 * center_penalty
        scored.append((score, (x, y, bw_, bh_, area)))

    scored.sort(reverse=True, key=lambda z: z[0])
    dx, dy, dw, dh, _ = scored[0][1]

    x0 = max(0, dx - drawing_region_padding)
    y0 = max(0, dy - drawing_region_padding)
    x1 = min(w, dx + dw + drawing_region_padding)
    y1 = min(h, dy + dh + drawing_region_padding)

    drawing_bbox = (x0, y0, x1 - x0, y1 - y0)

    metadata_boxes = []
    for _, (x, y, bw_, bh_, area) in scored[1:]:
        # keep blocks outside drawing region
        if x > x1 or x + bw_ < x0 or y > y1 or y + bh_ < y0:
            metadata_boxes.append((x, y, bw_, bh_))
        else:
            # partially outside margins
            if x < x0 or y < y0 or x + bw_ > x1 or y + bh_ > y1:
                metadata_boxes.append((x, y, bw_, bh_))

    metadata_boxes = _merge_boxes(metadata_boxes, pad=10)

    return PageRegions(drawing_bbox_img=drawing_bbox, metadata_bboxes_img=metadata_boxes)


def crop_to_bbox(image: Image.Image, bbox: Tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int]]:
    x, y, w, h = bbox
    cropped = image.crop((x, y, x + w, y + h))
    return cropped, (x, y)


def draw_page_regions(image: Image.Image, page_regions: PageRegions) -> Image.Image:
    img = np.array(image).copy()

    x, y, w, h = page_regions.drawing_bbox_img
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for bx, by, bw, bh in page_regions.metadata_bboxes_img:
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

    return Image.fromarray(img)