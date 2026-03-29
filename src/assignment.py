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


def _point_inside_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int], pad: int = 10) -> bool:
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
    max_distance: int = 180,
) -> float:
    score = 0.0

    inside_contour = _point_inside_contour(text_point_crop, room.contour)
    inside_bbox = _point_inside_bbox(text_point_crop, room.bbox, pad=10)

    if inside_contour:
        score += 100
    elif inside_bbox:
        score += 40
    else:
        return -1.0

    d = _distance(text_point_crop, room.centroid)
    if d <= max_distance:
        score += max(0, 25 - (d / max_distance) * 25)

    x, y, w, h = room.bbox
    room_box_area = w * h
    if room_box_area > 250000:
        score -= 25
    elif room_box_area > 150000:
        score -= 12

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
        crop_pt = (int(full_pt[0]) - int(crop_x), int(full_pt[1]) - int(crop_y))

        scored_rooms = []
        for room in rooms:
            s = score_text_room_pair(crop_pt, text, room)
            if s >= 0:
                scored_rooms.append((room, s))

        if not scored_rooms:
            continue

        scored_rooms.sort(key=lambda x: x[1], reverse=True)

        best_room, best_score = scored_rooms[0]

        # ambiguity check: skip if top-2 are too close
        if len(scored_rooms) > 1:
            second_score = scored_rooms[1][1]
            if abs(best_score - second_score) < 8:
                continue

        if best_score >= 85:
            best_room.assigned_labels.append(text)

    for room in rooms:
        if not room.assigned_labels:
            room.final_label = "UNNAMED_SPACE"
            room.confidence = 0.0
            room.evidence = []
            continue

        # if too many very different labels assigned, keep strongest only
        scored = sorted(room.assigned_labels, key=lambda t: t.score, reverse=True)
        best = scored[0]

        room.final_label = best.text
        room.confidence = min(max(best.score / 70.0, 0.0), 0.99)
        room.evidence = [t.text for t in scored[:3]]

    return rooms