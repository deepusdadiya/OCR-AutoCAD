import pandas as pd
from typing import List

from src.models import RoomCandidate, TextItem
from src.pdf_io import pdf_to_img_coords


def build_text_candidates_df(items: List[TextItem]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(items, start=1):
        rows.append(
            {
                "text_id": i,
                "text": t.text,
                "source": t.source,
                "region_type": t.region_type,
                "text_type": t.text_type,
                "label_category": t.label_category,
                "score": round(t.score, 2),
                "x0": round(t.x0, 2),
                "y0": round(t.y0, 2),
                "x1": round(t.x1, 2),
                "y1": round(t.y1, 2),
                "cx": round(t.cx, 2),
                "cy": round(t.cy, 2),
            }
        )
    return pd.DataFrame(rows)


def build_label_room_matches_df(
    label_candidates: List[TextItem],
    rooms: List[RoomCandidate],
    page_w: float,
    page_h: float,
    img_w: int,
    img_h: int,
    crop_offset: tuple[int, int],
) -> pd.DataFrame:
    crop_x, crop_y = crop_offset
    rows = []

    for i, text in enumerate(label_candidates, start=1):
        full_pt = pdf_to_img_coords(text.cx, text.cy, page_w, page_h, img_w, img_h)
        crop_pt = (int(full_pt[0]) - int(crop_x), int(full_pt[1]) - int(crop_y))

        best_room = None
        best_dist = None

        for room in rooms:
            rx, ry, rw, rh = room.bbox
            if rx <= crop_pt[0] <= rx + rw and ry <= crop_pt[1] <= ry + rh:
                d = ((crop_pt[0] - room.centroid[0]) ** 2 + (crop_pt[1] - room.centroid[1]) ** 2) ** 0.5
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_room = room

        rows.append(
            {
                "label_id": i,
                "extracted_label": text.text,
                "label_category": text.label_category,
                "score": round(text.score, 2),
                "pdf_cx": round(text.cx, 2),
                "pdf_cy": round(text.cy, 2),
                "crop_x": crop_pt[0],
                "crop_y": crop_pt[1],
                "matched_room_id": best_room.room_id if best_room else None,
                "matched_room_area_px": round(best_room.area_px, 2) if best_room else None,
                "matched_room_bbox": best_room.bbox if best_room else None,
            }
        )

    return pd.DataFrame(rows)


def build_rooms_df(rooms: List[RoomCandidate]) -> pd.DataFrame:
    rows = []
    for room in rooms:
        x, y, w, h = room.bbox
        rows.append(
            {
                "room_id": room.room_id,
                "predicted_name": room.final_label,
                "confidence": round(room.confidence, 3),
                "area_px": round(room.area_px, 2),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "centroid_x": room.centroid[0],
                "centroid_y": room.centroid[1],
                "evidence": " | ".join(room.evidence),
            }
        )
    return pd.DataFrame(rows)