from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

from src.models import RoomCandidate


def crop_image_by_ratio(
    image: Image.Image,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> tuple[Image.Image, tuple[int, int]]:
    w, h = image.size
    x0 = int(w * left_ratio)
    y0 = int(h * top_ratio)
    x1 = int(w * right_ratio)
    y1 = int(h * bottom_ratio)
    cropped = image.crop((x0, y0, x1, y1))
    return cropped, (x0, y0)


def detect_room_like_contours(
    image: Image.Image,
    min_contour_area: int = 1500,
    max_contour_area_ratio: float = 0.85,
) -> List[RoomCandidate]:
    """
    Detect enclosed room-like shapes from the rendered plan.
    This is a lightweight heuristic for POC only.
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # invert threshold so linework becomes white-ish for contour extraction
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.size[0] * image.size[1]
    rooms = []
    room_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        if area > img_area * max_contour_area_ratio:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # reject very thin / huge outer regions
        if w < 20 or h < 20:
            continue
        if w > image.size[0] * 0.95 and h > image.size[1] * 0.95:
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


def draw_rooms(image: Image.Image, rooms: List[RoomCandidate]) -> Image.Image:
    img = np.array(image).copy()
    for room in rooms:
        cnt = np.array(room.contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [cnt], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.circle(img, room.centroid, 3, (0, 255, 0), -1)
    return Image.fromarray(img)