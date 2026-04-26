from typing import List

from src.models import TextItem
from src.pdf_io import extract_pdf_lines


def extract_text_items(pdf_path: str, page_number: int = 0) -> List[TextItem]:
    # Generic vector-text first extractor using PDF line geometry and direction.
    return extract_pdf_lines(pdf_path, page_number=page_number)