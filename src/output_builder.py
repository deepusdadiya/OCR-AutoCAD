import pandas as pd
from typing import List

from src.models import RoomCandidate, TextItem


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
                "score": round(t.score, 2),
                "x0": round(t.x0, 2),
                "y0": round(t.y0, 2),
                "x1": round(t.x1, 2),
                "y1": round(t.y1, 2),
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