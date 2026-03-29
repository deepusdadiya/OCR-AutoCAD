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

# Text grouping
WORD_Y_TOL = 6.0
WORD_X_GAP_TOL = 22.0

# Text classification
MAX_TEXT_CHARS = 60
MAX_TEXT_WORDS = 6

# Geometry extraction
MIN_ROOM_AREA_PX = 10000
MAX_ROOM_AREA_RATIO = 0.35
WALL_BINARY_THRESHOLD = 210
MORPH_CLOSE_KERNEL = 3
MORPH_CLOSE_ITER = 2

# Assignment
TEXT_TO_ROOM_BBOX_PAD = 14
MAX_TEXT_ROOM_DISTANCE = 250

# OCR fallback
ENABLE_OCR_FALLBACK = False  # keep False for now