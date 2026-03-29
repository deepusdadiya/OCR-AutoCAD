import pandas as pd


def compare_with_expected(pred_df: pd.DataFrame, expected_csv_path: str) -> pd.DataFrame:
    expected_df = pd.read_csv(expected_csv_path)

    expected_df["expected_name_norm"] = expected_df.iloc[:, 0].astype(str).str.upper().str.strip()
    pred_df["predicted_name_norm"] = pred_df["predicted_name"].astype(str).str.upper().str.strip()

    comparison = expected_df.merge(
        pred_df,
        left_on="expected_name_norm",
        right_on="predicted_name_norm",
        how="left",
    )

    comparison["matched"] = comparison["predicted_name"].notna()
    return comparison