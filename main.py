from config import INPUT_PDF, EXPECTED_CSV, OUTPUT_DIR
from src.pipeline import run_pipeline


def main():
    result = run_pipeline(str(INPUT_PDF), str(EXPECTED_CSV))

    text_df = result["text_df"]
    rooms_df = result["rooms_df"]
    label_matches_df = result["label_matches_df"]
    final_df = result["final_df"]
    final_debug_df = result["final_debug_df"]
    comparison_df = result["comparison_df"]

    text_path = OUTPUT_DIR / "text_candidates.csv"
    rooms_path = OUTPUT_DIR / "predicted_rooms.csv"
    labels_path = OUTPUT_DIR / "label_matches.csv"
    final_path = OUTPUT_DIR / "final_output.csv"
    final_debug_path = OUTPUT_DIR / "final_output_debug.csv"

    text_df.to_csv(text_path, index=False)
    rooms_df.to_csv(rooms_path, index=False)
    label_matches_df.to_csv(labels_path, index=False)
    final_df.to_csv(final_path, index=False)
    final_debug_df.to_csv(final_debug_path, index=False)

    print(f"Saved: {text_path}")
    print(f"Saved: {rooms_path}")
    print(f"Saved: {labels_path}")
    print(f"Saved: {final_path}")
    print(f"Saved: {final_debug_path}")

    if comparison_df is not None:
        cmp_path = OUTPUT_DIR / "comparison.csv"
        comparison_df.to_csv(cmp_path, index=False)
        print(f"Saved: {cmp_path}")

    print("\nTop space-label candidates:")
    print(
        text_df[
            (text_df["text_type"] == "room_label") &
            (text_df["label_category"] == "space")
        ][["text", "region_type", "label_category", "score"]].head(40)
    )

    print("\nFinal output preview:")
    print(final_df)


if __name__ == "__main__":
    main()