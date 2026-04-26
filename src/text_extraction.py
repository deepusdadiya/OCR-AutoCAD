from typing import Any, Dict, List

from config import (
    ENABLE_OCR_FALLBACK,
    OCR_CONFIDENCE_THRESHOLD,
    OCR_MIN_ALPHA_ITEMS_TRIGGER,
    OCR_RENDER_DPI,
)
from src.models import TextItem
from src.ocr_fallback import extract_ocr_lines, ocr_backend_available
from src.pdf_io import extract_pdf_lines, get_page_size, render_pdf_page
from src.text_utils import has_alpha


def _alpha_item_count(items: List[TextItem]) -> int:
    return sum(1 for item in items if has_alpha(item.text))


def extract_text_payload(pdf_path: str, page_number: int = 0) -> Dict[str, Any]:
    vector_items = extract_pdf_lines(pdf_path, page_number=page_number)
    vector_alpha_count = _alpha_item_count(vector_items)

    payload: Dict[str, Any] = {
        "items": vector_items,
        "mode": "pdf_line",
        "ocr_available": ocr_backend_available(),
        "ocr_attempted": False,
        "vector_alpha_count": vector_alpha_count,
        "ocr_alpha_count": 0,
        "ocr_rotation_hits": {},
    }

    should_try_ocr = ENABLE_OCR_FALLBACK and vector_alpha_count < OCR_MIN_ALPHA_ITEMS_TRIGGER
    if not should_try_ocr:
        return payload

    page_w, page_h = get_page_size(pdf_path, page_number=page_number)
    image = render_pdf_page(pdf_path, page_number=page_number, dpi=OCR_RENDER_DPI)
    ocr_items, ocr_meta = extract_ocr_lines(
        image=image,
        page_w=page_w,
        page_h=page_h,
        page_number=page_number,
        confidence_threshold=OCR_CONFIDENCE_THRESHOLD,
    )

    ocr_alpha_count = _alpha_item_count(ocr_items)
    payload.update(ocr_meta)
    payload["ocr_attempted"] = True
    payload["ocr_alpha_count"] = ocr_alpha_count

    if ocr_alpha_count > vector_alpha_count:
        payload["items"] = ocr_items
        payload["mode"] = "ocr_line"
    elif vector_items:
        payload["mode"] = "pdf_line"
    else:
        payload["mode"] = "ocr_unavailable" if not payload["ocr_available"] else "pdf_line"

    return payload


def extract_text_items(pdf_path: str, page_number: int = 0) -> List[TextItem]:
    return extract_text_payload(pdf_path, page_number=page_number)["items"]
