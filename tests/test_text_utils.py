import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import TextItem  # noqa: E402
from src.text_classifier import attach_nearby_area_annotations, classify_text_item  # noqa: E402
from src.text_utils import clean_candidate_label_text, is_dimension_like, is_generic_metadata, is_spec_annotation  # noqa: E402


class TextUtilityTests(unittest.TestCase):
    def test_extracts_embedded_area_and_room_name(self) -> None:
        cleaned, area_value = clean_candidate_label_text("11.30 SQMT. POT WASH")
        self.assertEqual(cleaned, "POT WASH")
        self.assertEqual(area_value, 11.3)

        cleaned, area_value = clean_candidate_label_text("+ SERVICE BAR 18.33 SQMT.")
        self.assertEqual(cleaned, "SERVICE BAR")
        self.assertEqual(area_value, 18.33)

    def test_strips_trailing_size_annotation(self) -> None:
        cleaned, area_value = clean_candidate_label_text('F LOBBY 6\'-7"X8\'-8"')
        self.assertEqual(cleaned, "F LOBBY")
        self.assertIsNone(area_value)

        cleaned, area_value = clean_candidate_label_text('LOBBY 17\'X7\'-11"')
        self.assertEqual(cleaned, "LOBBY")
        self.assertIsNone(area_value)

    def test_detects_dimension_only_labels(self) -> None:
        self.assertTrue(is_dimension_like('24\'-8"X24\'-6"'))
        self.assertTrue(is_dimension_like('7\'-10"X9\''))
        self.assertTrue(is_dimension_like("11.14 SQMT."))

    def test_flags_spec_annotation(self) -> None:
        cleaned, _ = clean_candidate_label_text("1 NO. 1-WAY FLOW CASSETTE UNIT CAPACITY : 340 CFM / 0.6 TR")
        self.assertTrue(is_spec_annotation("1 NO. 1-WAY FLOW CASSETTE UNIT CAPACITY : 340 CFM / 0.6 TR", cleaned))

    def test_detects_domain_and_entry_exit_annotations(self) -> None:
        self.assertTrue(is_generic_metadata("HPGCONSULTING.COM"))
        self.assertTrue(is_generic_metadata("ENTRY/EXIT FOR AUDI./OBS./VIP"))

    def test_classifier_preserves_embedded_area_for_clean_label(self) -> None:
        item = TextItem(
            text="11.30 SQMT. POT WASH",
            x0=0,
            y0=0,
            x1=10,
            y1=10,
            region_type="drawing",
            font_size=10,
            dir_x=1,
            dir_y=0,
        )
        classified = classify_text_item(item, drawing_font_median=10.0)
        self.assertEqual(classified.text, "POT WASH")
        self.assertEqual(classified.text_type, "room_label")
        self.assertEqual(classified.embedded_area_value, 11.3)

    def test_attaches_nearby_area_annotation(self) -> None:
        label = TextItem(
            text="BALLROOM",
            x0=0,
            y0=0,
            x1=60,
            y1=10,
            region_type="drawing",
            text_type="room_label",
            font_size=10,
        )
        area_only = TextItem(
            text="431 SQM",
            x0=5,
            y0=12,
            x1=55,
            y1=20,
            region_type="drawing",
            text_type="unknown",
            font_size=8,
            embedded_area_value=431.0,
        )

        attach_nearby_area_annotations([label], [label, area_only])
        self.assertEqual(label.embedded_area_value, 431.0)


if __name__ == "__main__":
    unittest.main()
