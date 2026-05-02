from pathlib import Path

from config import EXPECTED_CSV, INPUT_PDF


def resolve_pdf_path(pdf_path: str | Path | None = None) -> Path:
    if pdf_path is None:
        return INPUT_PDF
    return Path(pdf_path).expanduser()


def resolve_expected_csv_path(
    pdf_path: str | Path,
    expected_path: str | Path | None = None,
) -> Path | None:
    if expected_path is not None:
        return Path(expected_path).expanduser()

    resolved_pdf_path = resolve_pdf_path(pdf_path)
    if resolved_pdf_path.resolve(strict=False) == INPUT_PDF.resolve(strict=False) and EXPECTED_CSV.exists():
        return EXPECTED_CSV

    return None
