import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.label_fusion import fuse_label_candidates  # noqa: E402
from src.models import TextItem  # noqa: E402


class LabelFusionTests(unittest.TestCase):
    def test_merges_vertical_observatory_label(self) -> None:
        items = [
            TextItem("LIFT FOR", 20, 20, 28, 60, source="pdf_line", page_number=0, block_no=1, line_no=0, font_size=8, dir_x=0, dir_y=1),
            TextItem("OBSERVATORY", 10, 15, 18, 75, source="pdf_line", page_number=0, block_no=1, line_no=1, font_size=8, dir_x=0, dir_y=1),
        ]

        fused = fuse_label_candidates(items)
        self.assertEqual([item.text for item in fused], ["LIFT FOR OBSERVATORY"])

    def test_merges_vertical_plumbing_shaft(self) -> None:
        items = [
            TextItem("PLUMBING", 100, 100, 106, 130, source="pdf_line", page_number=0, block_no=2, line_no=0, font_size=6, dir_x=0, dir_y=1),
            TextItem("SHAFT", 94, 108, 100, 124, source="pdf_line", page_number=0, block_no=2, line_no=1, font_size=6, dir_x=0, dir_y=1),
        ]

        fused = fuse_label_candidates(items)
        self.assertEqual([item.text for item in fused], ["PLUMBING SHAFT"])

    def test_does_not_merge_unrelated_vertical_labels(self) -> None:
        items = [
            TextItem("LIFT B5", 100, 100, 108, 130, source="pdf_line", page_number=0, block_no=3, line_no=0, font_size=8, dir_x=0, dir_y=1),
            TextItem("ELV", 150, 100, 156, 112, source="pdf_line", page_number=0, block_no=4, line_no=0, font_size=6, dir_x=0, dir_y=1),
            TextItem("RWS", 172, 99, 178, 112, source="pdf_line", page_number=0, block_no=5, line_no=0, font_size=6, dir_x=0, dir_y=1),
        ]

        fused = fuse_label_candidates(items)
        self.assertEqual(sorted(item.text for item in fused), ["ELV", "LIFT B5", "RWS"])


if __name__ == "__main__":
    unittest.main()
