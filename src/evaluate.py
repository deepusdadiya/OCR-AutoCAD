import re
from typing import Any, Dict, List

import pandas as pd

from src.text_utils import normalize_text


AREA_MATCH_TOLERANCE_RATIO = 0.05
AUTO_NUMBER_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+?)\s+#(?P<index>\d+)$")


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


def _comparison_group_key(name: str) -> str:
    normalized_name = normalize_text(name)
    match = AUTO_NUMBER_SUFFIX_PATTERN.fullmatch(normalized_name)
    if match is None:
        return normalized_name
    return match.group("base")


def _pairing_cost(expected_row: Dict[str, Any], predicted_row: Dict[str, Any] | None) -> float:
    if predicted_row is None:
        return 10_000.0

    expected_area = expected_row["expected_area"]
    predicted_area = predicted_row["predicted_area"]
    if pd.isna(expected_area) or pd.isna(predicted_area):
        area_cost = 1.0
    else:
        expected_value = float(expected_area)
        predicted_value = float(predicted_area)
        if expected_value == 0.0:
            area_cost = 0.0 if predicted_value == 0.0 else 1.0
        else:
            area_cost = abs(predicted_value - expected_value) / abs(expected_value)

    name_penalty = 0.0
    if expected_row["expected_name_norm"] != predicted_row["predicted_name_norm"]:
        name_penalty = 0.02

    return area_cost + name_penalty


def _best_group_assignment(
    expected_group: List[Dict[str, Any]],
    predicted_group: List[Dict[str, Any]],
) -> List[Dict[str, Any] | None]:
    best_cost = float("inf")
    best_assignment: List[Dict[str, Any] | None] = [None] * len(expected_group)

    def _search(
        expected_index: int,
        remaining_predicted: List[Dict[str, Any]],
        running_cost: float,
        current_assignment: List[Dict[str, Any] | None],
    ) -> None:
        nonlocal best_cost, best_assignment

        if running_cost >= best_cost:
            return

        if expected_index >= len(expected_group):
            best_cost = running_cost
            best_assignment = current_assignment.copy()
            return

        expected_row = expected_group[expected_index]

        for candidate_index, predicted_row in enumerate(remaining_predicted):
            current_assignment.append(predicted_row)
            _search(
                expected_index + 1,
                remaining_predicted[:candidate_index] + remaining_predicted[candidate_index + 1:],
                running_cost + _pairing_cost(expected_row, predicted_row),
                current_assignment,
            )
            current_assignment.pop()

        if len(remaining_predicted) < (len(expected_group) - expected_index):
            current_assignment.append(None)
            _search(
                expected_index + 1,
                remaining_predicted,
                running_cost + _pairing_cost(expected_row, None),
                current_assignment,
            )
            current_assignment.pop()

    _search(0, predicted_group, 0.0, [])
    return best_assignment


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
    expected_df["comparison_group"] = expected_df["expected_name"].map(_comparison_group_key)
    pred_df["comparison_group"] = pred_df["predicted_name"].map(_comparison_group_key)

    predicted_groups = {
        key: group.to_dict("records")
        for key, group in pred_df.groupby("comparison_group", sort=False)
    }

    comparison_rows: List[Dict[str, Any]] = []
    for _, expected_group_df in expected_df.groupby("comparison_group", sort=False):
        expected_group = expected_group_df.to_dict("records")
        predicted_group = predicted_groups.get(expected_group_df["comparison_group"].iloc[0], [])
        assignments = _best_group_assignment(expected_group, predicted_group)

        for expected_row, predicted_row in zip(expected_group, assignments):
            comparison_rows.append(
                {
                    "predicted_name": predicted_row["predicted_name"] if predicted_row is not None else pd.NA,
                    "expected_name": expected_row["expected_name"],
                    "predicted_area": predicted_row["predicted_area"] if predicted_row is not None else pd.NA,
                    "expected_area": expected_row["expected_area"],
                    "Name_Matched": (
                        predicted_row is not None
                        and predicted_row["comparison_group"] == expected_row["comparison_group"]
                    ),
                    "Area_Matched": _areas_match(
                        predicted_row["predicted_area"] if predicted_row is not None else pd.NA,
                        expected_row["expected_area"],
                    ),
                }
            )

    comparison = pd.DataFrame(comparison_rows)
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
