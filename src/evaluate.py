import pandas as pd

from src.text_utils import normalize_text


AREA_MATCH_TOLERANCE_RATIO = 0.05


def _resolve_area_column(df: pd.DataFrame) -> str | None:
    candidate_columns = [
        "Area (sqmm)",
        "Area",
        "area",
    ]
    for column in candidate_columns:
        if column in df.columns:
            return column
    return None


def _areas_match(predicted_area: float, expected_area: float) -> bool:
    if pd.isna(predicted_area) or pd.isna(expected_area):
        return False

    predicted = float(predicted_area)
    expected = float(expected_area)

    if expected == 0.0:
        return predicted == 0.0

    return abs(predicted - expected) / abs(expected) <= AREA_MATCH_TOLERANCE_RATIO


def compare_with_expected(pred_df: pd.DataFrame, expected_csv_path: str) -> pd.DataFrame:
    expected_df = pd.read_csv(expected_csv_path).copy()
    pred_df = pred_df.copy()

    expected_area_column = _resolve_area_column(expected_df)
    predicted_area_column = _resolve_area_column(pred_df)

    expected_df["expected_name"] = expected_df["Name"].astype(str)
    expected_df["expected_name_norm"] = expected_df["expected_name"].map(normalize_text)
    expected_df["expected_area"] = (
        pd.to_numeric(expected_df[expected_area_column], errors="coerce")
        if expected_area_column is not None
        else pd.Series(index=expected_df.index, dtype="float64")
    )

    pred_df["predicted_name"] = pred_df["Name"].astype(str)
    pred_df["predicted_name_norm"] = pred_df["predicted_name"].map(normalize_text)
    pred_df["predicted_area"] = (
        pd.to_numeric(pred_df[predicted_area_column], errors="coerce")
        if predicted_area_column is not None
        else pd.Series(index=pred_df.index, dtype="float64")
    )

    comparison = expected_df[
        ["expected_name", "expected_area", "expected_name_norm"]
    ].merge(
        pred_df[["predicted_name", "predicted_area", "predicted_name_norm"]],
        left_on="expected_name_norm",
        right_on="predicted_name_norm",
        how="left",
    )

    comparison["Name_Matched"] = comparison["predicted_name_norm"].notna()
    comparison["Area_Matched"] = comparison.apply(
        lambda row: _areas_match(row["predicted_area"], row["expected_area"]),
        axis=1,
    )
    return comparison[
        [
            "predicted_name",
            "expected_name",
            "predicted_area",
            "expected_area",
            "Name_Matched",
            "Area_Matched",
        ]
    ].copy()
