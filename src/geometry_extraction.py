from typing import List
import cv2
import numpy as np
from PIL import Image

from src.models import RoomCandidate


def detect_room_candidates(
    image: Image.Image,
    min_room_area_px: int = 6000,
    max_room_area_ratio: float = 0.50,
    wall_binary_threshold: int = 210,
    morph_close_kernel: int = 3,
    morph_close_iter: int = 2,
) -> List[RoomCandidate]:
    """
    Generic raster-based room candidate detection.
    No room-name hardcoding.
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, bw = cv2.threshold(gray, wall_binary_threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((morph_close_kernel, morph_close_kernel), np.uint8)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    img_area = image.size[0] * image.size[1]
    rooms = []
    room_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_room_area_px:
            continue
        if area > img_area * max_room_area_ratio:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 40:
            continue

        # Reject very thin / wall-like components
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 12:
            continue

        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        contour_pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]

        rooms.append(
            RoomCandidate(
                room_id=room_id,
                contour=contour_pts,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area_px=float(area),
            )
        )
        room_id += 1

    return rooms


def draw_room_candidates(image: Image.Image, rooms: List[RoomCandidate]) -> Image.Image:
    img = np.array(image).copy()

    for room in rooms:
        cnt = np.array(room.contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [cnt], True, (255, 0, 0), 2)
        cv2.circle(img, room.centroid, 3, (0, 255, 0), -1)

    return Image.fromarray(img)