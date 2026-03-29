import pandas as pd


def anchors_to_dataframe(anchors: list[dict]) -> pd.DataFrame:
    rows = []

    for i, item in enumerate(anchors, start=1):
        local_box = item.get("local_box")
        approx_area = None

        if local_box is not None:
            x0, y0, x1, y1 = local_box
            approx_area = (x1 - x0) * (y1 - y0)

        rows.append(
            {
                "label_id": i,
                "predicted_name": item["label"],
                "anchor_x": item["anchor"][0],
                "anchor_y": item["anchor"][1],
                "source": item["source"],
                "approx_local_area_px": approx_area,
            }
        )

    return pd.DataFrame(rows)