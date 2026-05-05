from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from src.models import TextItem
from src.text_utils import has_alpha, normalize_text


def _load_pytesseract() -> Any | None:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return None

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return None

    return pytesseract


@lru_cache(maxsize=1)
def _load_rapidocr() -> Any | None:
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:
        return None

    try:
        return RapidOCR()
    except Exception:
        return None


def ocr_backend_available() -> bool:
    return _load_pytesseract() is not None or _load_rapidocr() is not None


def _preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _rotation_variants(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    return [
        ("base", image),
        ("cw90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ("ccw90", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]


def _inverse_rotate_point(x: float, y: float, rotation: str, orig_w: int, orig_h: int) -> Tuple[float, float]:
    if rotation == "base":
        return x, y
    if rotation == "cw90":
        return y, orig_h - x
    if rotation == "ccw90":
        return orig_w - y, x
    raise ValueError(f"Unsupported rotation: {rotation}")


def _rotate_bbox_to_original(
    left: float,
    top: float,
    width: float,
    height: float,
    rotation: str,
    orig_w: int,
    orig_h: int,
) -> Tuple[float, float, float, float]:
    corners = [
        (left, top),
        (left + width, top),
        (left + width, top + height),
        (left, top + height),
    ]
    mapped = [_inverse_rotate_point(x, y, rotation, orig_w, orig_h) for x, y in corners]
    xs = [pt[0] for pt in mapped]
    ys = [pt[1] for pt in mapped]
    return min(xs), min(ys), max(xs), max(ys)


def _image_bbox_to_pdf_bbox(
    bbox: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    page_w: float,
    page_h: float,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        (x0 / img_w) * page_w,
        (y0 / img_h) * page_h,
        (x1 / img_w) * page_w,
        (y1 / img_h) * page_h,
    )


def _dedupe_lines(items: List[TextItem]) -> List[TextItem]:
    out: List[TextItem] = []
    for item in items:
        keep = True
        for existing in out:
            same_text = normalize_text(existing.text) == normalize_text(item.text)
            close_x = abs(existing.cx - item.cx) <= max(12.0, min(existing.width, item.width))
            close_y = abs(existing.cy - item.cy) <= max(12.0, min(existing.height, item.height))
            if same_text and close_x and close_y:
                keep = False
                break
        if keep:
            out.append(item)
    return out


def _orientation_from_bbox(pdf_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    width = max(pdf_bbox[2] - pdf_bbox[0], 1.0)
    height = max(pdf_bbox[3] - pdf_bbox[1], 1.0)
    if height > (width * 1.2):
        return 0.0, 1.0, width
    return 1.0, 0.0, height


def _extract_pytesseract_lines(
    pytesseract: Any,
    processed: np.ndarray,
    page_w: float,
    page_h: float,
    page_number: int,
    confidence_threshold: int,
) -> Tuple[List[TextItem], Dict[str, Any]]:
    img_h, img_w = processed.shape[:2]
    collected: List[TextItem] = []
    rotation_hits: Dict[str, int] = {}
    output_dict = pytesseract.Output.DICT

    for rotation, variant in _rotation_variants(processed):
        data = pytesseract.image_to_data(
            Image.fromarray(variant),
            output_type=output_dict,
            config="--oem 3 --psm 11",
        )

        groups: Dict[Tuple[int, int, int], List[Dict[str, float | str]]] = defaultdict(list)
        entry_count = len(data.get("text", []))
        for idx in range(entry_count):
            raw_text = str(data["text"][idx]).strip()
            if not raw_text:
                continue

            try:
                conf = float(data["conf"][idx])
            except Exception:
                conf = -1.0

            if conf < confidence_threshold:
                continue

            groups[(int(data["block_num"][idx]), int(data["par_num"][idx]), int(data["line_num"][idx]))].append(
                {
                    "text": raw_text,
                    "left": float(data["left"][idx]),
                    "top": float(data["top"][idx]),
                    "width": float(data["width"][idx]),
                    "height": float(data["height"][idx]),
                }
            )

        rotation_hits[rotation] = len(groups)

        for group_idx, words in enumerate(groups.values()):
            words = sorted(words, key=lambda word: float(word["left"]))
            text = normalize_text(" ".join(str(word["text"]) for word in words))
            if not text or not has_alpha(text):
                continue

            x0 = min(float(word["left"]) for word in words)
            y0 = min(float(word["top"]) for word in words)
            x1 = max(float(word["left"]) + float(word["width"]) for word in words)
            y1 = max(float(word["top"]) + float(word["height"]) for word in words)

            img_bbox = _rotate_bbox_to_original(x0, y0, x1 - x0, y1 - y0, rotation, img_w, img_h)
            pdf_bbox = _image_bbox_to_pdf_bbox(img_bbox, img_w, img_h, page_w, page_h)

            if rotation == "base":
                dir_x, dir_y = 1.0, 0.0
                font_size = max(pdf_bbox[3] - pdf_bbox[1], 1.0)
            else:
                dir_x, dir_y = 0.0, 1.0
                font_size = max(pdf_bbox[2] - pdf_bbox[0], 1.0)

            collected.append(
                TextItem(
                    text=text,
                    x0=float(pdf_bbox[0]),
                    y0=float(pdf_bbox[1]),
                    x1=float(pdf_bbox[2]),
                    y1=float(pdf_bbox[3]),
                    source="ocr_line",
                    block_no=-1,
                    line_no=group_idx,
                    word_no=0,
                    page_number=page_number,
                    font_size=float(font_size),
                    dir_x=dir_x,
                    dir_y=dir_y,
                )
            )

    return _dedupe_lines(collected), {
        "ocr_available": True,
        "ocr_attempted": True,
        "ocr_rotation_hits": rotation_hits,
    }


def _extract_rapidocr_lines(
    rapidocr_engine: Any,
    processed: np.ndarray,
    page_w: float,
    page_h: float,
    page_number: int,
    confidence_threshold: int,
) -> Tuple[List[TextItem], Dict[str, Any]]:
    img_h, img_w = processed.shape[:2]
    collected: List[TextItem] = []
    rotation_hits: Dict[str, int] = {}
    normalized_threshold = confidence_threshold / 100.0

    for rotation, variant in _rotation_variants(processed):
        try:
            result, _ = rapidocr_engine(variant)
        except Exception:
            result = None

        rows = result or []
        rotation_hits[rotation] = len(rows)

        for line_idx, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue

            box_points = row[0]
            text = normalize_text(str(row[1]).strip())
            try:
                confidence = float(row[2])
            except Exception:
                confidence = 0.0

            if not text or confidence < normalized_threshold or not has_alpha(text):
                continue

            xs = [float(point[0]) for point in box_points]
            ys = [float(point[1]) for point in box_points]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

            img_bbox = _rotate_bbox_to_original(x0, y0, x1 - x0, y1 - y0, rotation, img_w, img_h)
            pdf_bbox = _image_bbox_to_pdf_bbox(img_bbox, img_w, img_h, page_w, page_h)
            dir_x, dir_y, font_size = _orientation_from_bbox(pdf_bbox)

            collected.append(
                TextItem(
                    text=text,
                    x0=float(pdf_bbox[0]),
                    y0=float(pdf_bbox[1]),
                    x1=float(pdf_bbox[2]),
                    y1=float(pdf_bbox[3]),
                    source="ocr_line",
                    block_no=-1,
                    line_no=line_idx,
                    word_no=0,
                    page_number=page_number,
                    font_size=float(font_size),
                    dir_x=dir_x,
                    dir_y=dir_y,
                )
            )

    return _dedupe_lines(collected), {
        "ocr_available": True,
        "ocr_attempted": True,
        "ocr_rotation_hits": rotation_hits,
    }


def extract_ocr_lines(
    image: Image.Image,
    page_w: float,
    page_h: float,
    page_number: int,
    confidence_threshold: int = 55,
) -> Tuple[List[TextItem], Dict[str, Any]]:
    pytesseract = _load_pytesseract()
    processed = _preprocess_image(image)
    if pytesseract is not None:
        return _extract_pytesseract_lines(
            pytesseract,
            processed,
            page_w,
            page_h,
            page_number,
            confidence_threshold,
        )

    rapidocr_engine = _load_rapidocr()
    if rapidocr_engine is not None:
        return _extract_rapidocr_lines(
            rapidocr_engine,
            processed,
            page_w,
            page_h,
            page_number,
            confidence_threshold,
        )

    return [], {"ocr_available": False, "ocr_attempted": False, "ocr_rotation_hits": {}}
