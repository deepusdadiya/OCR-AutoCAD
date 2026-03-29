from typing import List
import cv2
import numpy as np

from src.models import RoomCandidate, TextItem
from src.pdf_io import pdf_to_img_coords


def _point_inside_contour(point: tuple[int, int], contour: list[tuple[int, int]]) -> bool:
    cnt = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
    px = float(point[0])
    py = float(point[1])
    return cv2.pointPolygonTest(cnt, (px, py), False) >= 0


def _point_inside_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int], pad: int = 14) -> bool:
    x, y, w, h = bbox
    px = float(point[0])
    py = float(point[1])
    return (x - pad) <= px <= (x + w + pad) and (y - pad) <= py <= (y + h + pad)


def _distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return ((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2) ** 0.5


def score_text_room_pair(
    text_point_crop: tuple[int, int],
    text_item: TextItem,
    room: RoomCandidate,
    bbox_pad: int = 14,
    max_distance: int = 250,
) -> float:
    score = 0.0

    if _point_inside_contour(text_point_crop, room.contour):
        score += 100
    elif _point_inside_bbox(text_point_crop, room.bbox, pad=bbox_pad):
        score += 55

    d = _distance(text_point_crop, room.centroid)
    if d <= max_distance:
        score += max(0, 35 - (d / max_distance) * 35)

    x, y, w, h = room.bbox
    px = float(text_point_crop[0])
    py = float(text_point_crop[1])

    if x <= px <= x + w and y <= py <= y + h:
        score += 15

    score += max(text_item.score, 0)

    return score


def assign_labels_to_rooms(
    rooms: List[RoomCandidate],
    text_items: List[TextItem],
    page_w: float,
    page_h: float,
    img_w: int,
    img_h: int,
    crop_offset: tuple[int, int],
) -> List[RoomCandidate]:
    if not rooms:
        return rooms

    crop_x, crop_y = crop_offset

    for text in text_items:
        full_pt = pdf_to_img_coords(text.cx, text.cy, page_w, page_h, img_w, img_h)

        crop_pt = (
            int(full_pt[0]) - int(crop_x),
            int(full_pt[1]) - int(crop_y),
        )

        best_room = None
        best_score = -1.0

        for room in rooms:
            s = score_text_room_pair(crop_pt, text, room)
            if s > best_score:
                best_score = s
                best_room = room

        if best_room is not None and best_score >= 70:
            best_room.assigned_labels.append(text)

    for room in rooms:
        if not room.assigned_labels:
            room.final_label = "UNNAMED_SPACE"
            room.confidence = 0.0
            room.evidence = []
            continue

        scored = sorted(room.assigned_labels, key=lambda t: t.score, reverse=True)
        best = scored[0]

        room.final_label = best.text
        room.confidence = min(max(best.score / 70.0, 0.0), 0.99)
        room.evidence = [t.text for t in scored[:5]]

    return rooms