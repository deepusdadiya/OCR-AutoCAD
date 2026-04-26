import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    manifest_path = REPO_ROOT / "data" / "regression_manifest.csv"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        print("No regression cases found.")
        return 1

    failures = []
    print("Text Extraction Regression")
    print("==========================")

    for row in rows:
        case_id = row["case_id"]
        pdf_path = REPO_ROOT / row["pdf_path"]
        expected_csv_path = REPO_ROOT / row["expected_csv_path"]
        min_match_rate = float(row["min_match_rate"])

        result = run_pipeline(str(pdf_path), str(expected_csv_path))
        comparison_df = result["comparison_df"]

        matched = int(comparison_df["matched"].sum()) if comparison_df is not None else 0
        expected_count = len(comparison_df) if comparison_df is not None else 0
        predicted_count = len(result["final_client_df"])
        match_rate = (matched / expected_count) if expected_count else 0.0

        print(f"{case_id}: matched {matched}/{expected_count} ({match_rate:.2%}), predicted {predicted_count}")

        if comparison_df is not None and (~comparison_df["matched"]).any():
            missing = comparison_df.loc[~comparison_df["matched"], "Name_x"].astype(str).tolist()
            print("  Missing:", ", ".join(missing))

        if match_rate < min_match_rate:
            failures.append((case_id, match_rate, min_match_rate))

    if failures:
        print("\nFailures:")
        for case_id, match_rate, min_match_rate in failures:
            print(f"  {case_id}: {match_rate:.2%} < required {min_match_rate:.2%}")
        return 1

    print("\nAll regression cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
