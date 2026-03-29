from typing import List
import cv2
import numpy as np
from PIL import Image

from src.models import RoomCandidate


def detect_room_candidates(
    image: Image.Image,
    min_room_area_px: int = 10000,
    max_room_area_ratio: float = 0.25,
    wall_binary_threshold: int = 210,
    morph_close_kernel: int = 3,
    morph_close_iter: int = 2,
) -> List[RoomCandidate]:
    """
    Generic room detection with hierarchy filtering.
    Rejects tiny fragments and giant merged parent regions.
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, bw = cv2.threshold(gray, wall_binary_threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((morph_close_kernel, morph_close_kernel), np.uint8)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    img_area = image.size[0] * image.size[1]

    candidates = []
    room_id = 1

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_room_area_px:
            continue
        if area > img_area * max_room_area_ratio:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 50:
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 8:
            continue

        # hierarchy info
        next_i, prev_i, child_i, parent_i = hierarchy[idx]

        # reject contours that have too many children -> usually merged outer blob
        child_count = 0
        c = child_i
        while c != -1:
            child_count += 1
            c = hierarchy[c][0]

        if child_count > 8:
            continue

        # prefer contours with a parent or with moderate compactness
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        compactness = 4 * np.pi * area / (perimeter * perimeter)

        if compactness < 0.005:
            continue

        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        contour_pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]

        candidates.append(
            RoomCandidate(
                room_id=room_id,
                contour=contour_pts,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area_px=float(area),
            )
        )
        room_id += 1

    # remove near-duplicate overlapping boxes
    filtered = []
    for room in sorted(candidates, key=lambda r: r.area_px):
        rx, ry, rw, rh = room.bbox
        keep = True
        for kept in filtered:
            kx, ky, kw, kh = kept.bbox

            inter_x0 = max(rx, kx)
            inter_y0 = max(ry, ky)
            inter_x1 = min(rx + rw, kx + kw)
            inter_y1 = min(ry + rh, ky + kh)

            if inter_x1 > inter_x0 and inter_y1 > inter_y0:
                inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                room_area = rw * rh
                kept_area = kw * kh
                overlap_ratio = inter_area / min(room_area, kept_area)

                if overlap_ratio > 0.85:
                    keep = False
                    break

        if keep:
            filtered.append(room)

    return filtered


def draw_room_candidates(image: Image.Image, rooms: List[RoomCandidate]) -> Image.Image:
    img = np.array(image).copy()

    for room in rooms:
        cnt = np.array(room.contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [cnt], True, (255, 0, 0), 2)
        cv2.circle(img, room.centroid, 3, (0, 255, 0), -1)
        cv2.putText(
            img,
            str(room.room_id),
            (room.centroid[0] + 4, room.centroid[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return Image.fromarray(img)