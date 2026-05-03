import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instance_export import build_final_client_instances_debug_df  # noqa: E402
from src.models import TextItem  # noqa: E402


class InstanceExportTests(unittest.TestCase):
    def test_area_column_stays_numeric_with_missing_values(self) -> None:
        items = [
            TextItem(text="ROOM A", x0=0, y0=0, x1=10, y1=10, area_value=12.5),
            TextItem(text="ROOM B", x0=20, y0=0, x1=30, y1=10, area_value=None, area_method="unresolved"),
        ]

        df = build_final_client_instances_debug_df(items)

        self.assertTrue(pd.api.types.is_float_dtype(df["Area (sqmm)"]))
        self.assertEqual(float(df.loc[df["Name"] == "ROOM A", "Area (sqmm)"].iloc[0]), 12.5)
        self.assertTrue(pd.isna(df.loc[df["Name"] == "ROOM B", "Area (sqmm)"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
