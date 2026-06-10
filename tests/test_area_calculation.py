import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import INPUT_PDF  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from src.area_calculation import (  # noqa: E402
    _area_pdf_to_sqm,
    _extract_scale_ratio,
    _infer_scale_ratio_from_embedded_areas,
    _infer_scale_ratio_from_size_annotations,
    extract_page_scale_ratio,
)
from src.models import TextItem  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


class AreaCalculationTests(unittest.TestCase):
    def test_extracts_scale_ratio_from_sample_pdf(self) -> None:
        scale_ratio = extract_page_scale_ratio(str(INPUT_PDF), page_number=0)
        self.assertEqual(scale_ratio, 100.0)

    def test_scale_parser_ignores_non_scale_ratios(self) -> None:
        self.assertIsNone(_extract_scale_ratio("RAMP SLOPE 1:12"))
        self.assertIsNone(_extract_scale_ratio("03/79"))
        self.assertIsNone(_extract_scale_ratio("REF.PIPE-1.3 TR 12.7/9.5"))

    def test_scale_parser_supports_architectural_format(self) -> None:
        self.assertEqual(_extract_scale_ratio('SCALE: 1/8" = 1\'-0"'), 96.0)

    def test_infers_scale_ratio_from_embedded_area(self) -> None:
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        expected_area_sqm = _area_pdf_to_sqm(polygon.area, 100.0)
        item = TextItem(
            text="ROOM",
            x0=2,
            y0=2,
            x1=8,
            y1=8,
            embedded_area_value=expected_area_sqm,
        )

        inferred_ratio = _infer_scale_ratio_from_embedded_areas([item], [polygon])
        self.assertIsNotNone(inferred_ratio)
        self.assertAlmostEqual(float(inferred_ratio), 100.0, places=3)

    def test_infers_scale_ratio_from_size_annotations(self) -> None:
        polygons = [
            Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
            Polygon([(20, 0), (30, 0), (30, 5), (20, 5)]),
            Polygon([(40, 0), (50, 0), (50, 5), (40, 5)]),
        ]
        size_items = [
            TextItem(text="SIZE-1000X500", x0=1, y0=1, x1=9, y1=4),
            TextItem(text="SIZE-1000X500", x0=21, y0=1, x1=29, y1=4),
            TextItem(text="SIZE-1000X500", x0=41, y0=1, x1=49, y1=4),
        ]

        inferred_ratio = _infer_scale_ratio_from_size_annotations(size_items, polygons)
        self.assertIsNotNone(inferred_ratio)
        self.assertAlmostEqual(float(inferred_ratio), 283.4645669, places=3)

    def test_resolves_exact_closed_room_areas(self) -> None:
        result = run_pipeline(str(INPUT_PDF), None)
        debug_df = result["final_client_debug_df"].set_index("Name")

        self.assertAlmostEqual(float(debug_df.loc["LIFT B1", "Area (sqmm)"]), 6.03, places=1)
        self.assertAlmostEqual(float(debug_df.loc["ELEC", "Area (sqmm)"]), 4.54, places=1)
        self.assertAlmostEqual(float(debug_df.loc["PLUMBING SHAFT", "Area (sqmm)"]), 1.63, places=1)

    def test_second_pass_resolves_open_spaces(self) -> None:
        result = run_pipeline(str(INPUT_PDF), None)
        debug_df = result["final_client_debug_df"].set_index("Name")

        self.assertEqual(debug_df.loc["F.TOILET", "area_method"], "vector_polygon_closed_gap")
        self.assertAlmostEqual(float(debug_df.loc["F.TOILET", "Area (sqmm)"]), 9.08, places=1)
        self.assertEqual(debug_df.loc["PANTRY", "area_method"], "vector_polygon_closed_gap")
        self.assertAlmostEqual(float(debug_df.loc["PANTRY", "Area (sqmm)"]), 4.12, places=1)
        self.assertEqual(debug_df.loc["M.TOILET", "area_method"], "vector_polygon_closed_gap")
        self.assertAlmostEqual(float(debug_df.loc["M.TOILET", "Area (sqmm)"]), 11.98, places=1)
        self.assertEqual(debug_df.loc["H.TOILET", "area_method"], "vector_polygon_closed_gap")
        self.assertAlmostEqual(float(debug_df.loc["H.TOILET", "Area (sqmm)"]), 4.15, places=1)
        self.assertEqual(debug_df.loc["PASSAGE #1", "area_method"], "vector_polygon_closed_gap")
        self.assertGreater(float(debug_df.loc["PASSAGE #1", "Area (sqmm)"]), 4.0)
        self.assertLess(float(debug_df.loc["PASSAGE #1", "Area (sqmm)"]), 5.0)
        self.assertEqual(debug_df.loc["PASSAGE #2", "area_method"], "vector_polygon_closed_gap")
        self.assertGreater(float(debug_df.loc["PASSAGE #2", "Area (sqmm)"]), 5.0)
        self.assertLess(float(debug_df.loc["PASSAGE #2", "Area (sqmm)"]), 5.5)
        self.assertEqual(debug_df.loc["LOBBY-3", "area_method"], "vector_polygon_closed_gap")
        self.assertGreater(float(debug_df.loc["LOBBY-3", "Area (sqmm)"]), 16.0)
        self.assertLess(float(debug_df.loc["LOBBY-3", "Area (sqmm)"]), 17.0)
        self.assertEqual(debug_df.loc["LOBBY-4", "area_method"], "vector_polygon_subtracted_shared_parent")
        self.assertGreater(float(debug_df.loc["LOBBY-4", "Area (sqmm)"]), 15.0)
        self.assertLess(float(debug_df.loc["LOBBY-4", "Area (sqmm)"]), 16.5)
        self.assertEqual(debug_df.loc["F LOBBY", "area_method"], "vector_polygon_subtracted_shared_parent")
        self.assertGreater(float(debug_df.loc["F LOBBY", "Area (sqmm)"]), 5.2)
        self.assertLess(float(debug_df.loc["F LOBBY", "Area (sqmm)"]), 5.8)
        self.assertEqual(debug_df.loc["STAIRCASE-1", "area_method"], "vector_polygon_subtracted_shared_parent")
        self.assertGreater(float(debug_df.loc["STAIRCASE-1", "Area (sqmm)"]), 32.0)
        self.assertLess(float(debug_df.loc["STAIRCASE-1", "Area (sqmm)"]), 32.7)
        self.assertEqual(debug_df.loc["LOBBY-1", "area_method"], "vector_open_residual_partition")
        self.assertGreater(float(debug_df.loc["LOBBY-1", "Area (sqmm)"]), 80.0)
        self.assertLess(float(debug_df.loc["LOBBY-1", "Area (sqmm)"]), 95.0)
        self.assertIn(debug_df.loc["OFFICE", "area_method"], {"vector_open_residual", "vector_open_residual_partition"})
        self.assertGreater(float(debug_df.loc["OFFICE", "Area (sqmm)"]), 1170.0)
        self.assertLess(float(debug_df.loc["OFFICE", "Area (sqmm)"]), 1195.0)

    def test_no_sample_labels_remain_unresolved(self) -> None:
        result = run_pipeline(str(INPUT_PDF), None)
        debug_df = result["final_client_debug_df"].set_index("Name")

        for name in ["A.H.U #1", "LOBBY-1", "LOBBY-2"]:
            with self.subTest(name=name):
                self.assertNotEqual(str(debug_df.loc[name, "Area (sqmm)"]).strip(), "")
                self.assertNotEqual(debug_df.loc[name, "area_method"], "unresolved")

    def test_area_assignments_store_geometry_references(self) -> None:
        result = run_pipeline(str(INPUT_PDF), None)
        items_by_name = {item.text: item for item in result["fused_label_candidates"]}

        self.assertTrue(items_by_name["LIFT B1"].area_geometry_refs)
        self.assertEqual(items_by_name["LIFT B1"].area_geometry_refs[0][0], "base")
        self.assertTrue(items_by_name["PANTRY"].area_geometry_refs)
        self.assertEqual(items_by_name["PANTRY"].area_geometry_refs[0][0], "door_aware")
        self.assertTrue(items_by_name["LOBBY-3"].area_geometry_refs)
        self.assertEqual(items_by_name["LOBBY-3"].area_geometry_refs[0][0], "door_aware")
        self.assertEqual(items_by_name["F LOBBY"].area_method, "vector_polygon_subtracted_shared_parent")
        self.assertTrue(items_by_name["F LOBBY"].area_geometry_shapes)
        self.assertEqual(items_by_name["STAIRCASE-1"].area_method, "vector_polygon_subtracted_shared_parent")
        self.assertTrue(items_by_name["STAIRCASE-1"].area_geometry_shapes)
        self.assertTrue(items_by_name["LOBBY-4"].area_geometry_refs)
        self.assertEqual(items_by_name["LOBBY-4"].area_method, "vector_polygon_subtracted_shared_parent")
        self.assertTrue(items_by_name["LOBBY-4"].area_geometry_shapes)


if __name__ == "__main__":
    unittest.main()
