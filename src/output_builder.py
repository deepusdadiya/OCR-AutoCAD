import pandas as pd
from typing import List

from src.models import TextItem


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
                "page_number": t.page_number,
                "font_size": round(t.font_size, 2),
                "orientation": t.orientation,
                "x0": round(t.x0, 2),
                "y0": round(t.y0, 2),
                "x1": round(t.x1, 2),
                "y1": round(t.y1, 2),
                "cx": round(t.cx, 2),
                "cy": round(t.cy, 2),
                "block_no": t.block_no,
                "line_no": t.line_no,
                "word_no": t.word_no,
            }
        )
    return pd.DataFrame(rows)