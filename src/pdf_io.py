import fitz
from PIL import Image
from typing import List, Tuple

from src.models import TextItem
from src.text_utils import normalize_text


def open_pdf(pdf_path: str) -> fitz.Document:
    return fitz.open(pdf_path)


def get_page_count(pdf_path: str) -> int:
    doc = open_pdf(pdf_path)
    page_count = doc.page_count
    doc.close()
    return page_count


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


def extract_pdf_lines(pdf_path: str, page_number: int = 0) -> List[TextItem]:
    doc = open_pdf(pdf_path)
    page = doc[page_number]
    text_dict = page.get_text("dict")
    doc.close()

    out = []
    for block_no, block in enumerate(text_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue

        for line_no, line in enumerate(block.get("lines", [])):
            spans = [span for span in line.get("spans", []) if str(span.get("text", "")).strip()]
            if not spans:
                continue

            text = normalize_text(" ".join(str(span.get("text", "")).strip() for span in spans))
            if not text:
                continue

            x0, y0, x1, y1 = line.get("bbox", spans[0].get("bbox"))
            dir_x, dir_y = line.get("dir", (1.0, 0.0))
            font_size = max(float(span.get("size", 0.0)) for span in spans)

            out.append(
                TextItem(
                    text=text,
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                    source="pdf_line",
                    block_no=int(block.get("number", block_no)),
                    line_no=int(line_no),
                    word_no=0,
                    page_number=page_number,
                    font_size=font_size,
                    dir_x=float(dir_x),
                    dir_y=float(dir_y),
                )
            )

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
