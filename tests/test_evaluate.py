import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import compare_with_expected  # noqa: E402


class EvaluateTests(unittest.TestCase):
    def test_compare_with_expected_returns_business_friendly_columns(self) -> None:
        pred_df = pd.DataFrame(
            [
                {"Name": "A.H.U #1", "Area (sqmm)": 24.85},
                {"Name": "CHW", "Area (sqmm)": 1.36},
            ]
        )
        expected_csv_path = REPO_ROOT / "data" / "expected" / "Room_areas - Sheet1 (1).csv"

        comparison_df = compare_with_expected(pred_df, str(expected_csv_path))

        self.assertEqual(
            list(comparison_df.columns),
            [
                "predicted_name",
                "expected_name",
                "predicted_area",
                "expected_area",
                "Name_Matched",
                "Area_Matched",
            ],
        )
        self.assertTrue(comparison_df.loc[0, "Name_Matched"])
        self.assertFalse(comparison_df.loc[0, "Area_Matched"])
        self.assertEqual(comparison_df.loc[0, "expected_name"], "A.H.U #1")
        self.assertEqual(comparison_df.loc[0, "predicted_name"], "A.H.U #1")
        self.assertAlmostEqual(float(comparison_df.loc[0, "expected_area"]), 15.3, places=2)
        self.assertAlmostEqual(float(comparison_df.loc[0, "predicted_area"]), 24.85, places=2)

    def test_area_matched_allows_five_percent_variation(self) -> None:
        pred_df = pd.DataFrame(
            [
                {"Name": "ROOM A", "Area (sqmm)": 104.9},
                {"Name": "ROOM B", "Area (sqmm)": 94.9},
            ]
        )
        expected_csv_path = REPO_ROOT / "tests" / "fixtures" / "expected_area_match.csv"

        comparison_df = compare_with_expected(pred_df, str(expected_csv_path))

        self.assertTrue(comparison_df.loc[0, "Area_Matched"])
        self.assertFalse(comparison_df.loc[1, "Area_Matched"])

    def test_duplicate_numbered_groups_match_best_area_pairing(self) -> None:
        pred_df = pd.DataFrame(
            [
                {"Name": "PASSAGE #1", "Area (sqmm)": 4.45},
                {"Name": "PASSAGE #2", "Area (sqmm)": 5.27},
            ]
        )
        expected_csv_path = REPO_ROOT / "tests" / "fixtures" / "expected_duplicate_pairing.csv"
        comparison_df = compare_with_expected(pred_df, str(expected_csv_path))

        self.assertEqual(comparison_df.loc[0, "predicted_name"], "PASSAGE #2")
        self.assertTrue(comparison_df.loc[0, "Name_Matched"])
        self.assertTrue(comparison_df.loc[0, "Area_Matched"])
        self.assertEqual(comparison_df.loc[1, "predicted_name"], "PASSAGE #1")
        self.assertTrue(comparison_df.loc[1, "Name_Matched"])
        self.assertTrue(comparison_df.loc[1, "Area_Matched"])


if __name__ == "__main__":
    unittest.main()
