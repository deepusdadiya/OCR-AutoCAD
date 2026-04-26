from config import INPUT_PDF, OUTPUT_DIR
from src.pipeline import run_pipeline


def main():
    result = run_pipeline(str(INPUT_PDF), None)

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

    text_path = OUTPUT_DIR / "text_candidates.csv"
    final_client_path = OUTPUT_DIR / "final_client_instances_output.csv"
    final_client_debug_path = OUTPUT_DIR / "final_client_instances_output_debug.csv"

    text_df.to_csv(text_path, index=False)
    final_client_df.to_csv(final_client_path, index=False)
    final_client_debug_df.to_csv(final_client_debug_path, index=False)

    print(f"Processed pages: {page_count}")
    print(f"Extraction modes: {', '.join(extraction_modes)}")
    print(f"Detected scales: {', '.join(scale_ratios)}")
    print(f"Resolved areas per page: {', '.join(str(x) for x in resolved_area_counts)}")
    print(f"Saved: {text_path}")
    print(f"Saved: {final_client_path}")
    print(f"Saved: {final_client_debug_path}")

    print("\nReconstructed text blocks preview:")
    print(text_df[["text", "region_type", "text_type", "label_category", "score"]].head(80))

    print("\nFinal client instance-level output preview:")
    print(final_client_df.head(60))


if __name__ == "__main__":
    main()
