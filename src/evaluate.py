import pandas as pd

from src.text_utils import normalize_text


def compare_with_expected(pred_df: pd.DataFrame, expected_csv_path: str) -> pd.DataFrame:
    expected_df = pd.read_csv(expected_csv_path).copy()
    pred_df = pred_df.copy()

    expected_df["expected_name_norm"] = expected_df["Name"].astype(str).map(normalize_text)
    pred_df["predicted_name_norm"] = pred_df["Name"].astype(str).map(normalize_text)

    comparison = expected_df.merge(
        pred_df[["Name", "predicted_name_norm"]],
        left_on="expected_name_norm",
        right_on="predicted_name_norm",
        how="left",
    )

    comparison["matched"] = comparison["predicted_name_norm"].notna()
    return comparison
