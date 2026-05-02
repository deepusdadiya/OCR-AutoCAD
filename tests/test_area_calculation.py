import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import INPUT_PDF  # noqa: E402
from src.area_calculation import extract_page_scale_ratio  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


class AreaCalculationTests(unittest.TestCase):
    def test_extracts_scale_ratio_from_sample_pdf(self) -> None:
        scale_ratio = extract_page_scale_ratio(str(INPUT_PDF), page_number=0)
        self.assertEqual(scale_ratio, 100.0)

    def test_resolves_exact_closed_room_areas(self) -> None:
        result = run_pipeline(str(INPUT_PDF), None)
        debug_df = result["final_client_debug_df"].set_index("Name")

        self.assertAlmostEqual(float(debug_df.loc["LIFT B1", "Area (sqmm)"]), 6.03, places=1)
        self.assertAlmostEqual(float(debug_df.loc["ELEC", "Area (sqmm)"]), 4.54, places=1)
        self.assertAlmostEqual(float(debug_df.loc["PLUMBING SHAFT", "Area (sqmm)"]), 1.63, places=1)

    def test_second_pass_resolves_open_spaces(self) -> None:
        result = run_pipeline(str(INPUT_PDF), None)
        debug_df = result["final_client_debug_df"].set_index("Name")

        self.assertEqual(debug_df.loc["F.TOILET", "area_method"], "vector_polygon_closed_gap")
        self.assertGreater(float(debug_df.loc["PANTRY", "Area (sqmm)"]), 3.0)
        self.assertLess(float(debug_df.loc["PANTRY", "Area (sqmm)"]), 5.0)
        self.assertEqual(debug_df.loc["STAIRCASE-1", "area_method"], "vector_fragment_cluster")
        self.assertGreater(float(debug_df.loc["STAIRCASE-1", "Area (sqmm)"]), 25.0)
        self.assertIn(debug_df.loc["OFFICE", "area_method"], {"vector_open_residual", "vector_open_residual_partition"})
        self.assertGreater(float(debug_df.loc["OFFICE", "Area (sqmm)"]), 1200.0)

    def test_no_sample_labels_remain_unresolved(self) -> None:
        result = run_pipeline(str(INPUT_PDF), None)
        debug_df = result["final_client_debug_df"].set_index("Name")

        for name in ["A.H.U #1", "LOBBY-1", "LOBBY-2"]:
            with self.subTest(name=name):
                self.assertNotEqual(str(debug_df.loc[name, "Area (sqmm)"]).strip(), "")
                self.assertNotEqual(debug_df.loc[name, "area_method"], "unresolved")


if __name__ == "__main__":
    unittest.main()
