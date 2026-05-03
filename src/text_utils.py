import re


GENERIC_METADATA_KEYWORDS = {
    "APPROVED",
    "ABOVE",
    "ARCHITECT",
    "BELOW",
    "CLIENT",
    "CONSULTANT",
    "COPYRIGHT",
    "DATE",
    "DOCUMENT",
    "DRAWING",
    "EMAIL",
    "ENGINEERING",
    "FUTURE",
    "GENERAL",
    "LINE",
    "NOTES",
    "PLAN",
    "PROJECT",
    "RAMP",
    "REVISION",
    "SCALE",
    "SHEET",
    "SIGNATURE",
    "SLOPE",
    "WIDTH",
    "WIDE",
}


DIMENSION_PATTERNS = [
    r"^\d+(\.\d+)?$",
    r"^\d+(\.\d+)?\s*(MM|CM|M|SQM|SQMM|FT|IN)$",
    r"^\d+(\.\d+)?\s*[Xx]\s*\d+(\.\d+)?$",
    r"^\d+(\.\d+)?\s*[Xx]\s*\d+(\.\d+)?\s*[Xx]\s*\d+(\.\d+)?$",
    r"^\d+(\.\d+)?\s*(SQM|SQMT|SQ\.?FT|SQFT|M2|MAı)\.?$",
    r"^\d+(?:\.\d+)?\s*[Xx]\s*\d+(?:\.\d+)?\s*(MM|CM|M)$",
]


INLINE_AREA_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(SQM|SQMT|SQ\.?FT|SQFT|M2|MAı)\.?\b")
LINEAR_SIZE_COMPONENT_PATTERN = r"(?:\d+\s*'\s*(?:-?\s*\d+\s*\")?|\d+\s*\"|\d+(?:\.\d+)?\s*(?:MM|CM|M)?)"
TRAILING_SIZE_PATTERN = re.compile(
    rf"\s+{LINEAR_SIZE_COMPONENT_PATTERN}(?:\s*[Xx]\s*{LINEAR_SIZE_COMPONENT_PATTERN})+$"
)
PAREN_SPEC_PATTERN = re.compile(r"\([^)]*\)")
DOMAIN_PATTERN = re.compile(
    r"\b(?:WWW\.|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|[A-Z0-9-]+\.(?:COM|CO|IN|NET|ORG|IO|AI))\b"
)
ENTRY_EXIT_PATTERN = re.compile(
    r"^(?:[A-Z]{1,4}\s+)?ENTRY(?:\s*(?:&|/|AND)\s*|\s+)EXIT(?:\s+FOR\b.*)?$|^(?:ENTRY|EXIT)$"
)
FRACTION_CODE_PATTERN = re.compile(r"^\d+/\d+[A-Z]{1,4}(?:\s*#?\s*\d+)?$")
SPEC_SIGNAL_KEYWORDS = (
    "CAPACITY",
    "CFM",
    "TR",
    "CUTOUT",
    "GRILLE",
    "REF.PIPE",
    "DRAIN PIPE",
    "CASSETTE",
    "PIPE-",
    "PIPE ",
)


def normalize_text(text: str) -> str:
    t = str(text).replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"\s+", " ", t).strip().upper()
    return t


def has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def alpha_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    alpha = sum(ch.isalpha() for ch in chars)
    return alpha / len(chars)


def punctuation_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    punct = sum(not ch.isalnum() for ch in chars)
    return punct / len(chars)


def token_count(text: str) -> int:
    return len([tok for tok in normalize_text(text).split(" ") if tok])


def clean_candidate_label_text(text: str) -> tuple[str, float | None]:
    t = normalize_text(text)
    embedded_area_value = None

    area_matches = INLINE_AREA_PATTERN.findall(t)
    if area_matches:
        embedded_area_value = float(area_matches[0][0])
        t = INLINE_AREA_PATTERN.sub(" ", t)

    t = PAREN_SPEC_PATTERN.sub(" ", t)
    t = re.sub(r"^[+\-*\/\\]+", " ", t)
    t = re.sub(r"\bSIZE\b.*$", " ", t)
    t = re.sub(r"\bCAPACITY\b.*$", " ", t)
    t = TRAILING_SIZE_PATTERN.sub("", t)
    t = re.sub(r"\s+", " ", t).strip(" .,:;+-_/\\")
    if t == "AREA":
        t = ""
    return normalize_text(t), embedded_area_value


def is_dimension_like(text: str) -> bool:
    t = normalize_text(text)
    if any(re.fullmatch(pattern, t) for pattern in DIMENSION_PATTERNS):
        return True

    if re.fullmatch(r"\d{3,}", t):
        return True

    if re.search(r"\b\d+(\.\d+)?\s*(MM|CM|M)\b", t):
        return True

    if re.fullmatch(
        rf"{LINEAR_SIZE_COMPONENT_PATTERN}(?:\s*[Xx]\s*{LINEAR_SIZE_COMPONENT_PATTERN})+",
        t,
    ):
        return True

    if re.fullmatch(
        r"\d+(?:\.\d+)?['\"]?(?:-\d+(?:\.\d+)?)?(\s*[Xx]\s*\d+(?:\.\d+)?['\"]?(?:-\d+(?:\.\d+)?)?)+",
        t,
    ):
        return True

    if FRACTION_CODE_PATTERN.fullmatch(t):
        return True

    return False


def is_generic_metadata(text: str) -> bool:
    t = normalize_text(text)
    if len(t) > 60:
        return True

    if DOMAIN_PATTERN.search(t):
        return True

    if ENTRY_EXIT_PATTERN.fullmatch(t):
        return True

    tokens = set(re.findall(r"[A-Z]+", t))
    return any(keyword in tokens for keyword in GENERIC_METADATA_KEYWORDS)


def is_spec_annotation(original_text: str, cleaned_text: str) -> bool:
    original = normalize_text(original_text)
    cleaned = normalize_text(cleaned_text)
    spec_hit_count = sum(1 for keyword in SPEC_SIGNAL_KEYWORDS if keyword in original)

    if spec_hit_count >= 2:
        return True

    if spec_hit_count >= 1 and re.search(r"\b(?:NO|NOS)\.?\b", original):
        return True

    if spec_hit_count >= 1 and re.match(r"^\d", original):
        return True

    if spec_hit_count >= 1 and cleaned in {"AREA", "CEILING CUTOUT"}:
        return True

    return False


def is_short_fragment(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return True

    if t in {"UP", "DN", "N", "S", "E", "W", "-", "."}:
        return True

    if re.fullmatch(r"\d+", t):
        return True

    if re.fullmatch(r"[A-Z](?:\s*#?\s*\d+)?", t):
        return True

    if re.fullmatch(r"[A-Z]\d{1,2}", t):
        return True

    return False


def similarity_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)
