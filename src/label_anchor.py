from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image

from src.models import TextItem


def pdf_text_to_crop_coords(
    text: TextItem,
    crop_offset: tuple[int, int],
    page_width: float,
    page_height: float,
    img_width: int,
    img_height: int,
) -> tuple[int, int]:
    offset_x, offset_y = crop_offset
    tx = int((text.cx / page_width) * img_width) - offset_x
    ty = int((text.cy / page_height) * img_height) - offset_y
    return tx, ty


def estimate_local_room_box(
    image: Image.Image,
    anchor_point: tuple[int, int],
    max_radius: int = 120,
) -> tuple[int, int, int, int] | None:
    """
    Very lightweight local region estimator around a label anchor.
    Expands around the text point until dark line boundaries are encountered.
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    cx, cy = anchor_point

    if not (0 <= cx < w and 0 <= cy < h):
        return None

    threshold = 210

    def walk_left():
        x = cx
        steps = 0
        while x > 1 and steps < max_radius:
            col = gray[max(0, cy - 3):min(h, cy + 4), x]
            if np.mean(col) < threshold:
                break
            x -= 1
            steps += 1
        return x

    def walk_right():
        x = cx
        steps = 0
        while x < w - 2 and steps < max_radius:
            col = gray[max(0, cy - 3):min(h, cy + 4), x]
            if np.mean(col) < threshold:
                break
            x += 1
            steps += 1
        return x

    def walk_up():
        y = cy
        steps = 0
        while y > 1 and steps < max_radius:
            row = gray[y, max(0, cx - 3):min(w, cx + 4)]
            if np.mean(row) < threshold:
                break
            y -= 1
            steps += 1
        return y

    def walk_down():
        y = cy
        steps = 0
        while y < h - 2 and steps < max_radius:
            row = gray[y, max(0, cx - 3):min(w, cx + 4)]
            if np.mean(row) < threshold:
                break
            y += 1
            steps += 1
        return y

    x0 = walk_left()
    x1 = walk_right()
    y0 = walk_up()
    y1 = walk_down()

    bw = x1 - x0
    bh = y1 - y0

    if bw < 20 or bh < 20:
        return None

    return (x0, y0, x1, y1)


def draw_anchor_boxes(
    image: Image.Image,
    anchors: List[dict],
) -> Image.Image:
    img = np.array(image).copy()

    for item in anchors:
        x, y = item["anchor"]
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

        if item["local_box"] is not None:
            x0, y0, x1, y1 = item["local_box"]
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)

        cv2.putText(
            img,
            item["label"][:20],
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return Image.fromarray(img)