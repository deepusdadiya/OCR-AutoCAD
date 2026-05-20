import argparse
from pathlib import Path

from config import INPUT_PDF, OUTPUT_DIR
from src.area_visualization import save_area_boundaries_visualization_pdf
from src.pipeline import run_pipeline
from src.runtime_inputs import resolve_expected_csv_path, resolve_pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the floor plan OCR pipeline.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=INPUT_PDF,
        help=f"Path to the input PDF. Defaults to the sample file at {INPUT_PDF}.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help="Optional expected CSV used for name comparison. If omitted, the sample expected CSV is only used for the sample PDF.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pdf_path = resolve_pdf_path(args.pdf)
    expected_csv_path = resolve_expected_csv_path(pdf_path, args.expected)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")
    if expected_csv_path is not None and not expected_csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {expected_csv_path}")

    result = run_pipeline(str(pdf_path), str(expected_csv_path) if expected_csv_path else None)

    page_count = result["page_count"]
    extraction_modes = [page["text_extraction_mode"] for page in result["page_results"]]
    resolved_area_counts = [page["resolved_area_count"] for page in result["page_results"]]
    scale_ratios = [
        f"1:{page['scale_ratio']:.0f}" if page["scale_ratio"] is not None else "unknown"
        for page in result["page_results"]
    ]
    text_df = result["text_df"]
    final_client_df = result["final_client_df"]
    final_client_debug_df = result["final_client_debug_df"]
    comparison_df = result["comparison_df"]

    text_path = OUTPUT_DIR / "text_candidates.csv"
    final_client_path = OUTPUT_DIR / "final_client_instances_output.csv"
    final_client_debug_path = OUTPUT_DIR / "final_client_instances_output_debug.csv"
    comparison_path = OUTPUT_DIR / "comparison.csv"

    text_df.to_csv(text_path, index=False)
    final_client_df.to_csv(final_client_path, index=False)
    final_client_debug_df.to_csv(final_client_debug_path, index=False)
    if comparison_df is not None:
        comparison_df.to_csv(comparison_path, index=False)
    visualization_path = save_area_boundaries_visualization_pdf(pdf_path, result["page_results"], OUTPUT_DIR)

    print(f"Input PDF: {pdf_path}")
    if expected_csv_path is not None:
        print(f"Expected CSV: {expected_csv_path}")
    print(f"Processed pages: {page_count}")
    print(f"Extraction modes: {', '.join(extraction_modes)}")
    print(f"Detected scales: {', '.join(scale_ratios)}")
    print(f"Resolved areas per page: {', '.join(str(x) for x in resolved_area_counts)}")
    print(f"Saved: {text_path}")
    print(f"Saved: {final_client_path}")
    print(f"Saved: {final_client_debug_path}")
    if visualization_path is not None:
        print(f"Saved: {visualization_path}")
    if comparison_df is not None:
        print(f"Saved: {comparison_path}")

    print("\nReconstructed text blocks preview:")
    print(text_df[["text", "region_type", "text_type", "label_category", "score"]].head(80))

    print("\nFinal client instance-level output preview:")
    print(final_client_df.head(60))


if __name__ == "__main__":
    main()
