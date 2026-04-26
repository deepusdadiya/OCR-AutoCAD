import re


GENERIC_METADATA_KEYWORDS = {
    "APPROVED",
    "ARCHITECT",
    "CLIENT",
    "CONSULTANT",
    "COPYRIGHT",
    "DATE",
    "DOCUMENT",
    "DRAWING",
    "EMAIL",
    "ENGINEERING",
    "GENERAL",
    "NOTES",
    "PROJECT",
    "REVISION",
    "SCALE",
    "SHEET",
    "SIGNATURE",
}


DIMENSION_PATTERNS = [
    r"^\d+(\.\d+)?$",
    r"^\d+(\.\d+)?\s*(MM|CM|M|SQM|SQMM|FT|IN)$",
    r"^\d+(\.\d+)?\s*[Xx]\s*\d+(\.\d+)?$",
    r"^\d+(\.\d+)?\s*[Xx]\s*\d+(\.\d+)?\s*[Xx]\s*\d+(\.\d+)?$",
]


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


def is_dimension_like(text: str) -> bool:
    t = normalize_text(text)
    if any(re.fullmatch(pattern, t) for pattern in DIMENSION_PATTERNS):
        return True

    if re.fullmatch(r"\d{3,}", t):
        return True

    if re.search(r"\b\d+(\.\d+)?\s*(MM|CM|M)\b", t):
        return True

    return False


def is_generic_metadata(text: str) -> bool:
    t = normalize_text(text)
    if len(t) > 60:
        return True

    return any(keyword in t for keyword in GENERIC_METADATA_KEYWORDS)


def is_short_fragment(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return True

    if t in {"UP", "DN", "N", "S", "E", "W", "-", "."}:
        return True

    if re.fullmatch(r"\d+", t):
        return True

    return False


def similarity_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)