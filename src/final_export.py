import pandas as pd
from typing import List, Dict, Any


def build_final_client_instances_df(resolved_payloads: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for payload in resolved_payloads:
        region = payload["region"]
        texts = payload["texts"]

        rows.append(
            {
                "Name": payload["instance_name"],
                "Area (sqmm)": "",   # keep blank until real geometric calibration is added
                "matched_region_id": region.room_id,
                "matched_region_area_px": round(region.area_px, 2),
                "matched_region_bbox": region.bbox,
                "evidence": " | ".join(t.text for t in texts),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(by=["Name"]).reset_index(drop=True)

    return df


def build_final_client_clean_df(resolved_payloads: List[Dict[str, Any]]) -> pd.DataFrame:
    df = build_final_client_instances_df(resolved_payloads).copy()
    return df[["Name", "Area (sqmm)"]]