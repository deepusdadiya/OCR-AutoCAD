from typing import List

from src.models import TextItem
from src.text_utils import normalize_text


def _clean_word(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return ""
    return t


def reconstruct_text_blocks(items: List[TextItem]) -> List[TextItem]:
    """
    Keep line-level extraction, normalize the text, and preserve geometry metadata.
    """
    if not items:
        return []

    cleaned = []
    for item in items:
        txt = _clean_word(item.text)
        if not txt:
            continue

        cleaned.append(
            TextItem(
                text=txt,
                x0=item.x0,
                y0=item.y0,
                x1=item.x1,
                y1=item.y1,
                source=item.source,
                region_type=item.region_type,
                text_type=item.text_type,
                label_category=item.label_category,
                score=item.score,
                block_no=item.block_no,
                line_no=item.line_no,
                word_no=item.word_no,
                page_number=item.page_number,
                font_size=item.font_size,
                dir_x=item.dir_x,
                dir_y=item.dir_y,
            )
        )

    return sorted(cleaned, key=lambda item: (item.page_number, item.y0, item.x0))


def deduplicate_blocks(items: List[TextItem]) -> List[TextItem]:
    seen = set()
    out = []
    for item in items:
        key = (item.text, round(item.cx, 1), round(item.cy, 1))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out