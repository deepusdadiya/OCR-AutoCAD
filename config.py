from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_PDF = BASE_DIR / "data" / "input" / "36th FLOOR - DG SHIPPING ARCH LAYOUT-Model.pdf"
EXPECTED_CSV = BASE_DIR / "data" / "expected" / "Room_areas - Sheet1 (1).csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rendering
RENDER_DPI = 220

# Page analysis
MIN_DRAWING_COMPONENT_AREA = 8000
DRAWING_REGION_PADDING = 20

# OCR fallback
ENABLE_OCR_FALLBACK = True
OCR_RENDER_DPI = 300
OCR_CONFIDENCE_THRESHOLD = 55
OCR_MIN_ALPHA_ITEMS_TRIGGER = 8
