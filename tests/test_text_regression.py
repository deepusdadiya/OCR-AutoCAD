import csv
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import run_pipeline  # noqa: E402


class TextRegressionTests(unittest.TestCase):
    def test_manifest_cases_meet_match_threshold(self) -> None:
        manifest_path = REPO_ROOT / "data" / "regression_manifest.csv"
        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            rows = list(csv.DictReader(manifest_file))
        self.assertTrue(rows, "Regression manifest is empty.")

        for row in rows:
            with self.subTest(case_id=row["case_id"]):
                pdf_path = REPO_ROOT / row["pdf_path"]
                expected_csv_path = REPO_ROOT / row["expected_csv_path"]
                min_match_rate = float(row["min_match_rate"])

                result = run_pipeline(str(pdf_path), str(expected_csv_path))
                comparison_df = result["comparison_df"]
                self.assertIsNotNone(comparison_df)

                matched = int(comparison_df["Name_Matched"].sum())
                expected_count = len(comparison_df)
                match_rate = matched / expected_count if expected_count else 0.0

                self.assertGreaterEqual(
                    match_rate,
                    min_match_rate,
                    f"{row['case_id']} matched {matched}/{expected_count} ({match_rate:.2%})",
                )


if __name__ == "__main__":
    unittest.main()
