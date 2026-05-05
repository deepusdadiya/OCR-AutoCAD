import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import TextItem  # noqa: E402
from src.text_extraction import extract_text_payload  # noqa: E402


class TextExtractionTests(unittest.TestCase):
    @patch("src.text_extraction.extract_ocr_lines")
    @patch("src.text_extraction.render_pdf_page")
    @patch("src.text_extraction.get_page_size")
    @patch("src.text_extraction.ocr_backend_available")
    @patch("src.text_extraction.extract_pdf_lines")
    def test_force_ocr_bypasses_alpha_threshold(
        self,
        mock_extract_pdf_lines,
        mock_ocr_backend_available,
        mock_get_page_size,
        mock_render_pdf_page,
        mock_extract_ocr_lines,
    ) -> None:
        mock_extract_pdf_lines.return_value = [
            TextItem(text=f"LABEL {index}", x0=0, y0=0, x1=1, y1=1)
            for index in range(10)
        ]
        mock_ocr_backend_available.return_value = True
        mock_get_page_size.return_value = (100.0, 100.0)
        mock_render_pdf_page.return_value = object()
        mock_extract_ocr_lines.return_value = (
            [TextItem(text="OCR LABEL", x0=0, y0=0, x1=1, y1=1)],
            {"ocr_available": True, "ocr_attempted": True, "ocr_rotation_hits": {"base": 1}},
        )

        payload = extract_text_payload("dummy.pdf", force_ocr=True)

        self.assertTrue(payload["ocr_attempted"])
        self.assertEqual(payload["mode"], "pdf_line")
        self.assertEqual(payload["ocr_alpha_count"], 1)


if __name__ == "__main__":
    unittest.main()
