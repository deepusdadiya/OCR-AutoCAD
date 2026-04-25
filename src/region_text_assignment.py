import math
from typing import List, Dict, Any
import cv2
import numpy as np

from src.models import RoomCandidate, TextItem
from src.pdf_io import pdf_to_img_coords


def _point_inside_contour(point: tuple[int, int], contour: list[tuple[int, int]]) -> bool:
    cnt = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(cnt, (float(point[0]), float(point[1])), False) >= 0


def _point_inside_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int], pad: int = 6) -> bool:
    x, y, w, h = bbox
    px, py = point
    return (x - pad) <= px <= (x + w + pad) and (y - pad) <= py <= (y + h + pad)


def _distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def collect_texts_for_regions(
    regions: List[RoomCandidate],
    text_items: List[TextItem],
    page_w: float,
    page_h: float,
    img_w: int,
    img_h: int,
    crop_offset: tuple[int, int],
) -> List[Dict[str, Any]]:
    """
    Assign text items locally to each region.
    Each text item can belong to the best matching region only.
    """
    crop_x, crop_y = crop_offset

    region_payloads: List[Dict[str, Any]] = []
    for region in regions:
        region_payloads.append(
            {
                "region": region,
                "texts": [],
            }
        )

    for text in text_items:
        full_pt = pdf_to_img_coords(text.cx, text.cy, page_w, page_h, img_w, img_h)
        crop_pt = (int(full_pt[0]) - int(crop_x), int(full_pt[1]) - int(crop_y))

        candidates = []
        for payload in region_payloads:
            region = payload["region"]

            inside_contour = _point_inside_contour(crop_pt, region.contour)
            inside_bbox = _point_inside_bbox(crop_pt, region.bbox, pad=8)
            if not inside_contour and not inside_bbox:
                continue

            d = _distance(crop_pt, region.centroid)

            x, y, w, h = region.bbox
            bbox_area = w * h

            score = 0.0
            if inside_contour:
                score += 100
            elif inside_bbox:
                score += 50

            score += max(0, 35 - min(d, 250) / 250 * 35)
            score += max(text.score, 0)

            # mild penalty for huge merged regions
            if bbox_area > 250000:
                score -= 20
            elif bbox_area > 150000:
                score -= 10

            candidates.append((payload, score))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1], reverse=True)

        # ambiguity rejection
        if len(candidates) > 1 and abs(candidates[0][1] - candidates[1][1]) < 6:
            continue

        best_payload = candidates[0][0]
        best_payload["texts"].append(text)

    return region_payloads