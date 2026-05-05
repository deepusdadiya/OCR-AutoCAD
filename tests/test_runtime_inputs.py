import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import EXPECTED_CSV, INPUT_PDF  # noqa: E402
from src.runtime_inputs import resolve_expected_csv_path, resolve_pdf_path  # noqa: E402


class RuntimeInputResolutionTests(unittest.TestCase):
    def test_defaults_to_sample_pdf(self) -> None:
        self.assertEqual(resolve_pdf_path(), INPUT_PDF)

    def test_uses_default_expected_for_sample_pdf(self) -> None:
        self.assertEqual(resolve_expected_csv_path(INPUT_PDF), EXPECTED_CSV)

    def test_does_not_apply_sample_expected_to_other_pdfs(self) -> None:
        other_pdf = REPO_ROOT / "data" / "input" / "another.pdf"
        self.assertIsNone(resolve_expected_csv_path(other_pdf))

    def test_explicit_expected_path_wins(self) -> None:
        explicit_expected = REPO_ROOT / "data" / "expected" / "custom.csv"
        self.assertEqual(resolve_expected_csv_path(INPUT_PDF, explicit_expected), explicit_expected)


if __name__ == "__main__":
    unittest.main()
