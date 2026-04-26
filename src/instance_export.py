import re
import pandas as pd
from typing import List

from src.models import TextItem
from src.text_utils import normalize_text


def _norm(text: str) -> str:
    return normalize_text(text)


def _needs_numbering(name: str) -> bool:
    t = _norm(name)

    # already specific, do not number
    if re.search(r"\b[A-Z]\d+\b", t):
        return False

    return True


def build_final_client_instances_df(label_candidates: List[TextItem]) -> pd.DataFrame:
    rows = []
    for item in label_candidates:
        rows.append(
            {
                "base_name": _norm(item.text),
                "score": round(item.score, 2),
                "page_number": item.page_number,
                "cx": round(item.cx, 2),
                "cy": round(item.cy, 2),
                "label_category": item.label_category,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Name", "Area (sqmm)"])

    df = df[df["base_name"].str.len() > 0].copy()
    df = df.sort_values(by=["base_name", "page_number", "cy", "cx"]).reset_index(drop=True)

    final_names = []
    for base_name, group in df.groupby("base_name", sort=False):
        idxs = list(group.index)

        if len(idxs) == 1 or not _needs_numbering(base_name):
            for idx in idxs:
                final_names.append((idx, base_name))
        else:
            for i, idx in enumerate(idxs, start=1):
                final_names.append((idx, f"{base_name} #{i}"))

    name_map = dict(final_names)
    df["Name"] = df.index.map(name_map)
    df["Area (sqmm)"] = ""

    return df[["Name", "Area (sqmm)"]].copy()


def build_final_client_instances_debug_df(label_candidates: List[TextItem]) -> pd.DataFrame:
    rows = []
    for item in label_candidates:
        rows.append(
            {
                "base_name": _norm(item.text),
                "score": round(item.score, 2),
                "page_number": item.page_number,
                "cx": round(item.cx, 2),
                "cy": round(item.cy, 2),
                "label_category": item.label_category,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Name", "Area (sqmm)", "page_number", "label_category", "score", "cx", "cy"])

    df = df.sort_values(by=["base_name", "page_number", "cy", "cx"]).reset_index(drop=True)

    final_names = []
    for base_name, group in df.groupby("base_name", sort=False):
        idxs = list(group.index)

        if len(idxs) == 1 or not _needs_numbering(base_name):
            for idx in idxs:
                final_names.append((idx, base_name))
        else:
            for i, idx in enumerate(idxs, start=1):
                final_names.append((idx, f"{base_name} #{i}"))

    name_map = dict(final_names)
    df["Name"] = df.index.map(name_map)
    df["Area (sqmm)"] = ""

    return df[["Name", "Area (sqmm)", "page_number", "label_category", "score", "cx", "cy"]].copy()
