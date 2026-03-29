from config import INPUT_PDF, EXPECTED_CSV, OUTPUT_DIR
from src.pipeline import run_pipeline


def main():
    result = run_pipeline(str(INPUT_PDF), str(EXPECTED_CSV))

    text_df = result["text_df"]
    rooms_df = result["rooms_df"]
    label_matches_df = result["label_matches_df"]
    comparison_df = result["comparison_df"]

    text_path = OUTPUT_DIR / "text_candidates.csv"
    rooms_path = OUTPUT_DIR / "predicted_rooms.csv"
    labels_path = OUTPUT_DIR / "label_matches.csv"

    text_df.to_csv(text_path, index=False)
    rooms_df.to_csv(rooms_path, index=False)
    label_matches_df.to_csv(labels_path, index=False)

    print(f"Saved: {text_path}")
    print(f"Saved: {rooms_path}")
    print(f"Saved: {labels_path}")

    if comparison_df is not None:
        cmp_path = OUTPUT_DIR / "comparison.csv"
        comparison_df.to_csv(cmp_path, index=False)
        print(f"Saved: {cmp_path}")

    print("\nTop room label candidates:")
    print(text_df[text_df["text_type"] == "room_label"][["text", "region_type", "score"]].head(40))

    print("\nLabel-first matches:")
    print(label_matches_df.head(25))

    print("\nPredicted rooms:")
    print(rooms_df.head(20))


if __name__ == "__main__":
    main()