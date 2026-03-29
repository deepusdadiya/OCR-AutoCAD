import fitz
from PIL import Image
from typing import List, Tuple

from src.models import TextItem


def open_pdf(pdf_path: str) -> fitz.Document:
    return fitz.open(pdf_path)


def get_page_size(pdf_path: str, page_number: int = 0) -> Tuple[float, float]:
    doc = open_pdf(pdf_path)
    page = doc[page_number]
    rect = page.rect
    doc.close()
    return rect.width, rect.height


def render_pdf_page(pdf_path: str, page_number: int = 0, dpi: int = 220) -> Image.Image:
    doc = open_pdf(pdf_path)
    page = doc[page_number]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def extract_pdf_words(pdf_path: str, page_number: int = 0) -> List[TextItem]:
    doc = open_pdf(pdf_path)
    page = doc[page_number]
    words = page.get_text("words")
    doc.close()

    out = []
    for w in words:
        x0, y0, x1, y1, text, *_ = w
        text = str(text).strip()
        if not text:
            continue
        out.append(TextItem(text=text, x0=x0, y0=y0, x1=x1, y1=y1, source="pdf_word"))
    return out


def pdf_to_img_coords(
    x: float,
    y: float,
    page_w: float,
    page_h: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int]:
    ix = int((x / page_w) * img_w)
    iy = int((y / page_h) * img_h)
    return ix, iy