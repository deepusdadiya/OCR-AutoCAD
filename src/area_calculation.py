import re
from collections import defaultdict
from fractions import Fraction
from math import exp, log, sqrt
from statistics import median
from typing import Any, Dict, List, Tuple

import fitz
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from config import (
    OCR_CONFIDENCE_THRESHOLD,
    OCR_RENDER_DPI,
    MIN_VECTOR_POLYGON_AREA_PDF,
    DOOR_CURVE_MIN_SIZE_PDF,
    DOOR_CURVE_MAX_SIZE_PDF,
    DOOR_GAP_LINE_TOLERANCE_PDF,
    DOOR_GAP_MAX_SPAN_PDF,
    DOOR_GAP_CURVE_MARGIN_PDF,
    DOOR_GAP_MIN_OVERLAP_PDF,
    MAX_EXACT_VECTOR_AREA_SQM,
    SECOND_PASS_DIRECT_MAX_AREA_SQM,
    OPEN_FRAGMENT_MAX_AREA_SQM,
    OPEN_FRAGMENT_NEARBY_DISTANCE_PDF,
    OPEN_FRAGMENT_MIN_ASSIGN_AREA_SQM,
    SPARSE_FRAGMENT_ASSIGN_RADIUS_PDF,
    SPARSE_FRAGMENT_MIN_AREA_SQM,
    OPEN_RESIDUAL_MIN_AREA_SQM,
    OPEN_RESIDUAL_MIN_DISTANCE_PDF,
    OPEN_RESIDUAL_DOMINANCE_RATIO,
    RESIDUAL_PARTITION_GRID_STEP_PDF,
    RESIDUAL_PARTITION_LOCAL_SHARE_POWER,
    RESIDUAL_PARTITION_FONT_POWER,
    RESIDUAL_PARTITION_FRAGMENT_SCALE_PDF,
    RESIDUAL_PARTITION_MAX_DISTANCE_PDF,
    NETWORKED_CLOSED_GAP_MIN_NEIGHBORS,
    NETWORKED_CLOSED_GAP_AREA_RATIO,
    TINY_FRAGMENT_MAX_AREA_SQM,
    FRAGMENT_CLUSTER_SIGNAL_RADIUS_PDF,
    FRAGMENT_CLUSTER_ASSIGN_RADIUS_PDF,
    FRAGMENT_CLUSTER_COMPETITION_RADIUS_PDF,
    MIN_TINY_FRAGMENT_COUNT,
    MIN_TINY_FRAGMENT_AREA_SQM,
    AMBIGUOUS_FRAGMENT_NEARBY_LABELS,
    AMBIGUOUS_FRAGMENT_AREA_SQM,
    AMBIGUOUS_FRAGMENT_FILL_RATIO,
    SHARED_FRAGMENT_FALLBACK_FACTOR,
    SHARED_FRAGMENT_FALLBACK_MAX_EXISTING_AREA_SQM,
)
from src.models import TextItem
from src.ocr_fallback import extract_ocr_lines, ocr_backend_available
from src.pdf_io import extract_pdf_lines, get_page_size, render_pdf_page
from src.text_utils import normalize_text


SCALE_PATTERNS = [
    re.compile(r"\bSCALE\b[^0-9]*(1(?:\.0+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\b"),
    re.compile(r"^(1(?:\.0+)?)\s*[:/]\s*(\d+(?:\.\d+)?)$"),
]
ARCHITECTURAL_SCALE_PATTERN = re.compile(
    r"(?P<draw>\d+(?:/\d+)?)\s*\"\s*=\s*(?P<feet>\d+)\s*'\s*(?:[- ]\s*(?P<inches>\d+)\s*\")?"
)
SCALE_KEYWORD_PATTERN = re.compile(r"\bSCALE\b")
SIZE_ANNOTATION_PATTERN = re.compile(
    r"(?:SIZE[:=-]?\s*)?(\d+(?:\.\d+)?)\s*(MM|CM|M)?\s*[Xx]\s*(\d+(?:\.\d+)?)\s*(MM|CM|M)?"
)


def _ratio_from_numeric_scale(left: float, right: float) -> float | None:
    if left <= 0 or right <= 0:
        return None
    ratio = right / left
    if 5 <= ratio <= 5000:
        return ratio
    return None


def _extract_architectural_scale_ratio(text: str) -> float | None:
    upper_text = text.upper()
    match = ARCHITECTURAL_SCALE_PATTERN.search(upper_text)
    if not match:
        return None

    draw_inches = float(Fraction(match.group("draw")))
    feet = int(match.group("feet"))
    inches = int(match.group("inches") or 0)
    real_inches = (feet * 12) + inches
    return _ratio_from_numeric_scale(draw_inches, float(real_inches))


def _extract_scale_ratio(text: str) -> float | None:
    lines = [line for line in text.splitlines() if line.strip()]
    return _extract_scale_ratio_from_lines(lines)


def _extract_scale_ratio_from_lines(lines: List[str]) -> float | None:
    normalized_lines = [normalize_text(line) for line in lines if normalize_text(line)]

    for index, line in enumerate(normalized_lines):
        if not SCALE_KEYWORD_PATTERN.search(line):
            continue

        context_lines = normalized_lines[max(0, index - 1): min(len(normalized_lines), index + 2)]
        for context_line in context_lines:
            ratio = _extract_architectural_scale_ratio(context_line)
            if ratio is not None:
                return ratio

            for pattern in SCALE_PATTERNS:
                match = pattern.search(context_line)
                if match is None:
                    continue
                ratio = _ratio_from_numeric_scale(float(match.group(1)), float(match.group(2)))
                if ratio is not None:
                    return ratio

    for line in normalized_lines:
        ratio = _extract_architectural_scale_ratio(line)
        if ratio is not None:
            return ratio

        for pattern in SCALE_PATTERNS:
            match = pattern.fullmatch(line)
            if match is None:
                continue
            ratio = _ratio_from_numeric_scale(float(match.group(1)), float(match.group(2)))
            if ratio is not None:
                return ratio

    return None


def extract_page_scale_ratio(pdf_path: str, page_number: int) -> float | None:
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    text = page.get_text("text")
    doc.close()

    ratio = _extract_scale_ratio(text)
    if ratio is not None:
        return ratio

    pdf_lines = extract_pdf_lines(pdf_path, page_number=page_number)
    ratio = _extract_scale_ratio_from_lines([item.text for item in pdf_lines])
    if ratio is not None:
        return ratio

    if not ocr_backend_available():
        return None

    page_w, page_h = get_page_size(pdf_path, page_number=page_number)
    image = render_pdf_page(pdf_path, page_number=page_number, dpi=OCR_RENDER_DPI)
    ocr_items, _ = extract_ocr_lines(
        image=image,
        page_w=page_w,
        page_h=page_h,
        page_number=page_number,
        confidence_threshold=OCR_CONFIDENCE_THRESHOLD,
    )
    return _extract_scale_ratio_from_lines([item.text for item in ocr_items])


def _quad_segments(quad: Any) -> List[LineString]:
    points = [
        (float(quad.ul.x), float(quad.ul.y)),
        (float(quad.ur.x), float(quad.ur.y)),
        (float(quad.lr.x), float(quad.lr.y)),
        (float(quad.ll.x), float(quad.ll.y)),
        (float(quad.ul.x), float(quad.ul.y)),
    ]
    return [LineString([points[i], points[i + 1]]) for i in range(len(points) - 1)]


def _is_grayish_color(color: Tuple[float, float, float] | None) -> bool:
    if not color:
        return False
    return (max(color) - min(color)) < 0.05


def _drawing_is_probable_symbol_layer(drawing: Dict[str, Any]) -> bool:
    width = float(drawing.get("width") or 0.0)
    return width <= 0.01 and _is_grayish_color(drawing.get("color"))


def _curve_bbox(curve_item: Any) -> Tuple[float, float, float, float]:
    points = [
        (float(curve_item[1].x), float(curve_item[1].y)),
        (float(curve_item[2].x), float(curve_item[2].y)),
        (float(curve_item[3].x), float(curve_item[3].y)),
        (float(curve_item[4].x), float(curve_item[4].y)),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _is_probable_door_curve(curve_item: Any, drawing: Dict[str, Any]) -> bool:
    min_x, min_y, max_x, max_y = _curve_bbox(curve_item)
    width = max_x - min_x
    height = max_y - min_y
    if min(width, height) < DOOR_CURVE_MIN_SIZE_PDF:
        return False
    if max(width, height) > DOOR_CURVE_MAX_SIZE_PDF:
        return False

    aspect_ratio = max(width, height) / max(min(width, height), 1e-6)
    if aspect_ratio > 2.5:
        return False

    return _drawing_is_probable_symbol_layer(drawing) or float(drawing.get("width") or 0.0) <= 0.01


def _normalize_segment_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    rounded_start = (round(float(start[0]), 2), round(float(start[1]), 2))
    rounded_end = (round(float(end[0]), 2), round(float(end[1]), 2))
    return (rounded_start, rounded_end) if rounded_start <= rounded_end else (rounded_end, rounded_start)


def _segment_tuples_from_page(
    page: fitz.Page,
    include_symbol_segments: bool,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]], List[Tuple[float, float, float, float]]]:
    segment_tuples: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    door_curve_bboxes: List[Tuple[float, float, float, float]] = []

    for drawing in page.get_drawings():
        items = drawing.get("items", [])
        symbol_layer = _drawing_is_probable_symbol_layer(drawing)

        for item in items:
            if item[0] == "c" and _is_probable_door_curve(item, drawing):
                door_curve_bboxes.append(_curve_bbox(item))

        if symbol_layer and not include_symbol_segments:
            continue

        for item in items:
            kind = item[0]
            if kind == "l":
                p1, p2 = item[1], item[2]
                segment_tuples.append(
                    _normalize_segment_points(
                        (float(p1.x), float(p1.y)),
                        (float(p2.x), float(p2.y)),
                    )
                )
            elif kind == "qu":
                for segment in _quad_segments(item[1]):
                    start, end = list(segment.coords)
                    segment_tuples.append(_normalize_segment_points(start, end))

    return segment_tuples, door_curve_bboxes


def _merge_axis_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    ordered = sorted(intervals)
    merged: List[List[float]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + DOOR_GAP_LINE_TOLERANCE_PDF:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def _build_axis_interval_index(
    segment_tuples: List[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> Tuple[Dict[float, List[Tuple[float, float]]], Dict[float, List[Tuple[float, float]]]]:
    vertical: Dict[float, List[Tuple[float, float]]] = defaultdict(list)
    horizontal: Dict[float, List[Tuple[float, float]]] = defaultdict(list)

    for start, end in segment_tuples:
        if abs(start[0] - end[0]) <= DOOR_GAP_LINE_TOLERANCE_PDF:
            x = round((start[0] + end[0]) / 2.0, 2)
            y0, y1 = sorted((start[1], end[1]))
            vertical[x].append((y0, y1))
        elif abs(start[1] - end[1]) <= DOOR_GAP_LINE_TOLERANCE_PDF:
            y = round((start[1] + end[1]) / 2.0, 2)
            x0, x1 = sorted((start[0], end[0]))
            horizontal[y].append((x0, x1))

    return (
        {x: _merge_axis_intervals(intervals) for x, intervals in vertical.items()},
        {y: _merge_axis_intervals(intervals) for y, intervals in horizontal.items()},
    )


def _detect_door_gap_bridge_segments(
    segment_tuples: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    door_curve_bboxes: List[Tuple[float, float, float, float]],
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if not segment_tuples or not door_curve_bboxes:
        return []

    vertical, horizontal = _build_axis_interval_index(segment_tuples)
    bridge_segments: set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()

    for min_x, min_y, max_x, max_y in door_curve_bboxes:
        expanded_x0 = min_x - DOOR_GAP_CURVE_MARGIN_PDF
        expanded_y0 = min_y - DOOR_GAP_CURVE_MARGIN_PDF
        expanded_x1 = max_x + DOOR_GAP_CURVE_MARGIN_PDF
        expanded_y1 = max_y + DOOR_GAP_CURVE_MARGIN_PDF

        for x, intervals in vertical.items():
            if x < expanded_x0 or x > expanded_x1:
                continue

            for (first_start, first_end), (second_start, second_end) in zip(intervals, intervals[1:]):
                gap_start = first_end
                gap_end = second_start
                gap_span = gap_end - gap_start
                if gap_span <= DOOR_GAP_LINE_TOLERANCE_PDF or gap_span > DOOR_GAP_MAX_SPAN_PDF:
                    continue

                gap_mid = (gap_start + gap_end) / 2.0
                if gap_mid < expanded_y0 or gap_mid > expanded_y1:
                    continue

                overlap = min(gap_end, expanded_y1) - max(gap_start, expanded_y0)
                if overlap < DOOR_GAP_MIN_OVERLAP_PDF:
                    continue

                bridge_segments.add(_normalize_segment_points((x, gap_start), (x, gap_end)))

        for y, intervals in horizontal.items():
            if y < expanded_y0 or y > expanded_y1:
                continue

            for (first_start, first_end), (second_start, second_end) in zip(intervals, intervals[1:]):
                gap_start = first_end
                gap_end = second_start
                gap_span = gap_end - gap_start
                if gap_span <= DOOR_GAP_LINE_TOLERANCE_PDF or gap_span > DOOR_GAP_MAX_SPAN_PDF:
                    continue

                gap_mid = (gap_start + gap_end) / 2.0
                if gap_mid < expanded_x0 or gap_mid > expanded_x1:
                    continue

                overlap = min(gap_end, expanded_x1) - max(gap_start, expanded_x0)
                if overlap < DOOR_GAP_MIN_OVERLAP_PDF:
                    continue

                bridge_segments.add(_normalize_segment_points((gap_start, y), (gap_end, y)))

    return sorted(bridge_segments)


def _curve_closure_segments(curve_item: Any) -> List[LineString]:
    start = (float(curve_item[1].x), float(curve_item[1].y))
    ctrl1 = (float(curve_item[2].x), float(curve_item[2].y))
    ctrl2 = (float(curve_item[3].x), float(curve_item[3].y))
    end = (float(curve_item[4].x), float(curve_item[4].y))

    if abs(start[0] - end[0]) > 1.0 and abs(start[1] - end[1]) > 1.0:
        hinge_a = (end[0], start[1])
        hinge_b = (start[0], end[1])
        score_a = abs(ctrl1[1] - start[1]) + abs(ctrl2[0] - end[0])
        score_b = abs(ctrl1[0] - start[0]) + abs(ctrl2[1] - end[1])
        hinge = hinge_a if score_a <= score_b else hinge_b
        return [
            LineString(
                [
                    (round(start[0], 2), round(start[1], 2)),
                    (round(hinge[0], 2), round(hinge[1], 2)),
                ]
            ),
            LineString(
                [
                    (round(hinge[0], 2), round(hinge[1], 2)),
                    (round(end[0], 2), round(end[1], 2)),
                ]
            ),
        ]

    return [
        LineString(
            [
                (round(start[0], 2), round(start[1], 2)),
                (round(end[0], 2), round(end[1], 2)),
            ]
        )
    ]


def build_page_vector_polygons(
    pdf_path: str,
    page_number: int,
    close_door_arcs: bool = False,
    use_door_gap_bridges: bool = False,
) -> List[Polygon]:
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    if use_door_gap_bridges:
        segment_tuples, door_curve_bboxes = _segment_tuples_from_page(page, include_symbol_segments=False)
        segment_tuples.extend(_detect_door_gap_bridge_segments(segment_tuples, door_curve_bboxes))
    else:
        segments: List[LineString] = []
        for drawing in page.get_drawings():
            for item in drawing.get("items", []):
                kind = item[0]
                if kind == "l":
                    p1, p2 = item[1], item[2]
                    segments.append(
                        LineString(
                            [
                                (round(float(p1.x), 2), round(float(p1.y), 2)),
                                (round(float(p2.x), 2), round(float(p2.y), 2)),
                            ]
                        )
                    )
                elif kind == "qu":
                    for segment in _quad_segments(item[1]):
                        segments.append(
                            LineString(
                                [
                                    (round(segment.coords[0][0], 2), round(segment.coords[0][1], 2)),
                                    (round(segment.coords[1][0], 2), round(segment.coords[1][1], 2)),
                                ]
                            )
                        )
                elif kind == "c" and close_door_arcs:
                    segments.extend(_curve_closure_segments(item))
    doc.close()

    if use_door_gap_bridges:
        if not segment_tuples:
            return []
        segments = [LineString([start, end]) for start, end in sorted(set(segment_tuples))]
    elif not segments:
        return []
    polygons = [
        polygon
        for polygon in polygonize(unary_union(segments))
        if polygon.area >= MIN_VECTOR_POLYGON_AREA_PDF
    ]
    return polygons


def _area_pdf_to_sqm(area_pdf: float, scale_ratio: float) -> float:
    mm_per_pdf_unit = (25.4 / 72.0) * scale_ratio
    return (area_pdf * (mm_per_pdf_unit**2)) / 1_000_000.0


def _dimension_to_mm(value: str, unit: str | None) -> float:
    normalized_unit = (unit or "MM").upper()
    numeric_value = float(value)
    if normalized_unit == "M":
        return numeric_value * 1000.0
    if normalized_unit == "CM":
        return numeric_value * 10.0
    return numeric_value


def _infer_scale_ratio_from_embedded_areas(
    label_candidates: List[TextItem],
    polygons: List[Polygon],
    polygon_tree: STRtree | None = None,
) -> float | None:
    inferred_ratios: List[float] = []
    pdf_unit_to_mm = 25.4 / 72.0

    for item in label_candidates:
        if item.embedded_area_value is None or item.embedded_area_value <= 0:
            continue

        containing_indices = _find_containing_polygon_indices(Point(item.cx, item.cy), polygons, polygon_tree)
        if not containing_indices:
            continue

        area_pdf = polygons[containing_indices[0]].area
        if area_pdf <= 0:
            continue

        mm_per_pdf_unit = sqrt((item.embedded_area_value * 1_000_000.0) / area_pdf)
        ratio = mm_per_pdf_unit / pdf_unit_to_mm
        if 5.0 <= ratio <= 5000.0:
            inferred_ratios.append(ratio)

    if not inferred_ratios:
        return None

    center_ratio = median(inferred_ratios)
    clustered = [ratio for ratio in inferred_ratios if 0.75 <= (ratio / center_ratio) <= 1.25]
    if clustered:
        return float(median(clustered))

    return float(center_ratio)


def _infer_scale_ratio_from_size_annotations(
    size_items: List[TextItem],
    polygons: List[Polygon],
) -> float | None:
    if not size_items or not polygons:
        return None

    candidate_rows: List[Tuple[float, float, float]] = []
    pdf_unit_to_mm = 25.4 / 72.0

    for item in size_items:
        match = SIZE_ANNOTATION_PATTERN.search(normalize_text(item.text))
        if match is None:
            continue

        real_dims_mm = sorted(
            [
                _dimension_to_mm(match.group(1), match.group(2)),
                _dimension_to_mm(match.group(3), match.group(4)),
            ],
            reverse=True,
        )

        item_point = Point(item.cx, item.cy)
        local_candidates: List[Tuple[float, float, float]] = []
        for polygon in polygons:
            min_x, min_y, max_x, max_y = polygon.bounds
            pdf_dims = sorted([max_x - min_x, max_y - min_y], reverse=True)
            if pdf_dims[1] <= 1.0:
                continue

            distance = item_point.distance(polygon)
            if distance > 2.0:
                continue

            real_ratio = real_dims_mm[0] / max(real_dims_mm[1], 1.0)
            pdf_ratio = pdf_dims[0] / max(pdf_dims[1], 1e-6)
            aspect_error = abs(log(real_ratio / pdf_ratio))
            if aspect_error > 0.35:
                continue

            scale_w = (real_dims_mm[0] / pdf_dims[0]) / pdf_unit_to_mm
            scale_h = (real_dims_mm[1] / pdf_dims[1]) / pdf_unit_to_mm
            if not (20.0 <= scale_w <= 1000.0 and 20.0 <= scale_h <= 1000.0):
                continue
            if abs(log(scale_w / scale_h)) > 0.3:
                continue

            local_candidates.append((distance, aspect_error, (scale_w + scale_h) / 2.0))

        if not local_candidates:
            continue

        candidate_rows.append(min(local_candidates, key=lambda row: (row[0], row[1])))

    if len(candidate_rows) < 3:
        return None

    scales = [row[2] for row in candidate_rows]
    best_cluster: List[Tuple[float, float, float]] = []
    for center_scale in scales:
        cluster = [
            row
            for row in candidate_rows
            if 0.82 <= (row[2] / center_scale) <= 1.22
        ]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
            continue
        if len(cluster) == len(best_cluster) and cluster:
            cluster_distance = median(row[0] for row in cluster)
            best_distance = median(row[0] for row in best_cluster) if best_cluster else float("inf")
            if cluster_distance < best_distance:
                best_cluster = cluster

    if len(best_cluster) < 3:
        return None

    return float(median(row[2] for row in best_cluster))


def _find_containing_polygon_indices(
    point: Point,
    polygons: List[Polygon],
    polygon_tree: STRtree | None = None,
) -> List[int]:
    candidate_indices = range(len(polygons))
    if polygon_tree is not None and polygons:
        candidate_indices = polygon_tree.query(point.buffer(0.5))

    return sorted(
        [int(index) for index in candidate_indices if polygons[int(index)].buffer(0.5).contains(point)],
        key=lambda index: polygons[index].area,
    )


def _smallest_containing_area_sqm(
    point: Point,
    polygons: List[Polygon],
    polygon_areas_sqm: List[float],
    polygon_tree: STRtree | None = None,
) -> Tuple[int | None, float | None]:
    containing = _find_containing_polygon_indices(point, polygons, polygon_tree)
    if not containing:
        return None, None
    smallest_idx = containing[0]
    return smallest_idx, polygon_areas_sqm[smallest_idx]


def _find_parent_polygon_index(
    point: Point,
    polygons: List[Polygon],
    polygon_areas_sqm: List[float],
    polygon_tree: STRtree | None = None,
) -> int | None:
    for index in _find_containing_polygon_indices(point, polygons, polygon_tree):
        area_sqm = polygon_areas_sqm[index]
        if MAX_EXACT_VECTOR_AREA_SQM < area_sqm < 2000.0:
            return index
    return None


def _assign_exact_area(
    item: TextItem,
    area_sqm: float,
    method: str,
    confidence: float,
    geometry_refs: List[Tuple[str, int]] | None = None,
) -> None:
    item.area_value = round(area_sqm, 2)
    item.area_method = method
    item.area_confidence = confidence
    item.area_geometry_refs = list(geometry_refs or [])


def _assign_text_embedded_area(item: TextItem) -> None:
    item.area_value = round(item.embedded_area_value or 0.0, 2)
    item.area_method = "text_embedded_area"
    item.area_confidence = 0.6
    item.area_geometry_refs = []


def _assign_unresolved(item: TextItem) -> None:
    item.area_value = None
    item.area_method = "unresolved"
    item.area_confidence = 0.0
    item.area_geometry_refs = []


def _polygon_fill_ratio(polygon: Polygon) -> float:
    min_x, min_y, max_x, max_y = polygon.bounds
    bbox_area = (max_x - min_x) * (max_y - min_y)
    if bbox_area <= 0:
        return 0.0
    return polygon.area / bbox_area


def _build_fragment_indices(polygon_areas_sqm: List[float]) -> List[int]:
    return [
        index
        for index, area_sqm in enumerate(polygon_areas_sqm)
        if 0.05 <= area_sqm <= OPEN_FRAGMENT_MAX_AREA_SQM
    ]


def _nearest_fragment_distance(
    point: Point,
    polygons: List[Polygon],
    fragment_indices: List[int],
) -> float:
    if not fragment_indices:
        return float("inf")
    return min(point.distance(polygons[index]) for index in fragment_indices)


def _find_fragment_candidate_indices(
    polygons: List[Polygon],
    polygon_areas_sqm: List[float],
    blocked_items: List[TextItem],
) -> List[int]:
    blocked_points = [Point(item.cx, item.cy) for item in blocked_items]
    candidate_indices: List[int] = []

    for index, polygon in enumerate(polygons):
        area_sqm = polygon_areas_sqm[index]
        if area_sqm < 0.05 or area_sqm > OPEN_FRAGMENT_MAX_AREA_SQM:
            continue
        if any(polygon.buffer(0.5).contains(point) for point in blocked_points):
            continue
        candidate_indices.append(index)

    return candidate_indices


def _assign_sparse_isolated_fragment_areas(
    unresolved_items: List[TextItem],
    all_items: List[TextItem],
    polygons: List[Polygon],
    polygon_areas_sqm: List[float],
    polygon_tree: STRtree | None = None,
) -> None:
    if not unresolved_items:
        return

    unresolved_by_parent: Dict[int, List[TextItem]] = defaultdict(list)
    for item in unresolved_items:
        point = Point(item.cx, item.cy)
        parent_index = _find_parent_polygon_index(point, polygons, polygon_areas_sqm, polygon_tree)
        if parent_index is None:
            continue
        unresolved_by_parent[parent_index].append(item)

    tiny_fragment_indices = [
        index
        for index, area_sqm in enumerate(polygon_areas_sqm)
        if 0.05 <= area_sqm <= TINY_FRAGMENT_MAX_AREA_SQM
    ]
    if not tiny_fragment_indices:
        return

    item_points = {id(item): Point(item.cx, item.cy) for item in all_items}

    for item in unresolved_items:
        if item.area_value is not None:
            continue

        point = item_points[id(item)]
        parent_index = _find_parent_polygon_index(point, polygons, polygon_areas_sqm, polygon_tree)
        if parent_index is None or len(unresolved_by_parent.get(parent_index, [])) != 1:
            continue

        nearby_regular = any(point.distance(polygons[index]) <= OPEN_FRAGMENT_NEARBY_DISTANCE_PDF for index in tiny_fragment_indices)
        if nearby_regular:
            continue

        total_area_sqm = 0.0
        assigned_indices: List[int] = []
        for index in tiny_fragment_indices:
            polygon = polygons[index]
            distance = point.distance(polygon)
            if distance > SPARSE_FRAGMENT_ASSIGN_RADIUS_PDF:
                continue

            competitions = sorted(
                [
                    (other_point.distance(polygon), other_item)
                    for other_item in all_items
                    for other_point in [item_points[id(other_item)]]
                    if other_point.distance(polygon) <= SPARSE_FRAGMENT_ASSIGN_RADIUS_PDF
                ],
                key=lambda pair: pair[0],
            )
            if not competitions or id(competitions[0][1]) != id(item):
                continue

            total_area_sqm += polygon_areas_sqm[index]
            assigned_indices.append(index)

        if total_area_sqm >= SPARSE_FRAGMENT_MIN_AREA_SQM:
            _assign_exact_area(
                item,
                total_area_sqm,
                "vector_sparse_fragment_fallback",
                0.24,
                geometry_refs=[("enhanced", index) for index in assigned_indices],
            )


def _reopen_networked_closed_gap_labels(
    label_candidates: List[TextItem],
    polygons: List[Polygon],
    polygon_areas_sqm: List[float],
    polygon_tree: STRtree | None = None,
) -> None:
    for item in label_candidates:
        if item.area_method != "vector_polygon_closed_gap":
            continue

        point = Point(item.cx, item.cy)
        polygon_index, polygon_area_sqm = _smallest_containing_area_sqm(point, polygons, polygon_areas_sqm, polygon_tree)
        if polygon_index is None or polygon_area_sqm is None:
            continue

        neighbor_indices = [
            index
            for index, polygon in enumerate(polygons)
            if index != polygon_index
            and 0.05 <= polygon_areas_sqm[index] <= OPEN_FRAGMENT_MAX_AREA_SQM
            and polygons[polygon_index].distance(polygon) <= 2.0
        ]
        neighbor_area_sqm = sum(polygon_areas_sqm[index] for index in neighbor_indices)

        if (
            len(neighbor_indices) >= NETWORKED_CLOSED_GAP_MIN_NEIGHBORS
            and neighbor_area_sqm >= (polygon_area_sqm * NETWORKED_CLOSED_GAP_AREA_RATIO)
        ):
            _assign_unresolved(item)


def _assign_direct_second_pass_areas(
    label_candidates: List[TextItem],
    enhanced_polygons: List[Polygon],
    enhanced_polygon_areas_sqm: List[float],
    enhanced_polygon_tree: STRtree | None = None,
) -> None:
    for item in label_candidates:
        if item.area_value is not None:
            continue

        point = Point(item.cx, item.cy)
        polygon_index, area_sqm = _smallest_containing_area_sqm(
            point,
            enhanced_polygons,
            enhanced_polygon_areas_sqm,
            enhanced_polygon_tree,
        )
        if polygon_index is None or area_sqm is None:
            continue
        if area_sqm > SECOND_PASS_DIRECT_MAX_AREA_SQM:
            continue

        _assign_exact_area(
            item,
            area_sqm,
            "vector_polygon_closed_gap",
            0.72,
            geometry_refs=[("door_aware", polygon_index)],
        )


def _identify_fragment_rich_labels(
    active_items: List[TextItem],
    candidate_indices: List[int],
    polygons: List[Polygon],
    polygon_areas_sqm: List[float],
) -> List[TextItem]:
    ranked_items: List[Tuple[float, TextItem]] = []

    for item in active_items:
        point = Point(item.cx, item.cy)
        nearby_areas = [
            polygon_areas_sqm[index]
            for index in candidate_indices
            if point.distance(polygons[index]) <= FRAGMENT_CLUSTER_SIGNAL_RADIUS_PDF
        ]
        tiny_fragment_areas = [area_sqm for area_sqm in nearby_areas if area_sqm <= TINY_FRAGMENT_MAX_AREA_SQM]

        if len(tiny_fragment_areas) < MIN_TINY_FRAGMENT_COUNT:
            continue
        if sum(tiny_fragment_areas) < MIN_TINY_FRAGMENT_AREA_SQM:
            continue

        ranked_items.append((sum(nearby_areas), item))

    ranked_items.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked_items]


def _preassign_fragment_rich_clusters(
    active_items: List[TextItem],
    candidate_indices: List[int],
    polygons: List[Polygon],
    polygon_areas_sqm: List[float],
) -> set[int]:
    rich_items = _identify_fragment_rich_labels(active_items, candidate_indices, polygons, polygon_areas_sqm)
    if not rich_items:
        return set()

    active_points = {id(item): Point(item.cx, item.cy) for item in active_items}
    claimed_indices: set[int] = set()

    for rich_item in rich_items:
        point = active_points[id(rich_item)]
        total_area_sqm = 0.0
        assigned_indices: List[int] = []

        for index in candidate_indices:
            if index in claimed_indices:
                continue

            polygon = polygons[index]
            if point.distance(polygon) > FRAGMENT_CLUSTER_ASSIGN_RADIUS_PDF:
                continue

            competitions = sorted(
                [
                    (other_point.distance(polygon), other_item)
                    for other_item in active_items
                    for other_point in [active_points[id(other_item)]]
                    if other_point.distance(polygon) <= FRAGMENT_CLUSTER_COMPETITION_RADIUS_PDF
                ],
                key=lambda pair: pair[0],
            )
            if not competitions:
                continue

            target_distance = next(
                (distance for distance, other_item in competitions if id(other_item) == id(rich_item)),
                None,
            )
            if target_distance is None:
                continue

            best_distance, best_item = competitions[0]
            if id(best_item) == id(rich_item) or target_distance <= ((best_distance * 1.12) + 4.0):
                total_area_sqm += polygon_areas_sqm[index]
                claimed_indices.add(index)
                assigned_indices.append(index)

        if total_area_sqm >= OPEN_FRAGMENT_MIN_ASSIGN_AREA_SQM:
            _assign_exact_area(
                rich_item,
                total_area_sqm,
                "vector_fragment_cluster",
                0.48,
                geometry_refs=[("enhanced", index) for index in assigned_indices],
            )

    return claimed_indices


def _assign_fragment_partition_areas(
    unresolved_items: List[TextItem],
    resolved_items: List[TextItem],
    polygons: List[Polygon],
    polygon_areas_sqm: List[float],
) -> None:
    candidate_indices = _find_fragment_candidate_indices(polygons, polygon_areas_sqm, resolved_items)
    if not candidate_indices:
        return

    claimed_indices = _preassign_fragment_rich_clusters(
        unresolved_items,
        candidate_indices,
        polygons,
        polygon_areas_sqm,
    )

    active_items = [item for item in unresolved_items if item.area_value is None]
    item_area_totals: Dict[int, float] = defaultdict(float)
    item_geometry_refs: Dict[int, set[int]] = defaultdict(set)
    deferred_indices: set[int] = set()

    for index in candidate_indices:
        if index in claimed_indices:
            continue

        polygon = polygons[index]
        area_sqm = polygon_areas_sqm[index]
        fill_ratio = _polygon_fill_ratio(polygon)
        nearby: List[Tuple[TextItem, float]] = []

        for item in active_items:
            point = Point(item.cx, item.cy)
            distance = point.distance(polygon)
            if polygon.buffer(0.5).contains(point) or distance <= OPEN_FRAGMENT_NEARBY_DISTANCE_PDF:
                nearby.append((item, distance))

        if not nearby:
            continue

        if (
            len(nearby) >= AMBIGUOUS_FRAGMENT_NEARBY_LABELS
            and (area_sqm > AMBIGUOUS_FRAGMENT_AREA_SQM or fill_ratio < AMBIGUOUS_FRAGMENT_FILL_RATIO)
        ):
            deferred_indices.add(index)
            continue

        if len(nearby) == 1:
            winner = nearby[0][0]
            item_area_totals[id(winner)] += area_sqm
            item_geometry_refs[id(winner)].add(index)
            continue

        weights = [(item, 1.0 / max(distance, 5.0)) for item, distance in nearby]
        total_weight = sum(weight for _, weight in weights)
        if total_weight <= 0:
            continue

        for item, weight in weights:
            item_area_totals[id(item)] += area_sqm * (weight / total_weight)
            item_geometry_refs[id(item)].add(index)

    for item in active_items:
        area_sqm = item_area_totals.get(id(item), 0.0)
        if area_sqm < OPEN_FRAGMENT_MIN_ASSIGN_AREA_SQM:
            continue
        _assign_exact_area(
            item,
            area_sqm,
            "vector_open_fragment_partition",
            0.42,
            geometry_refs=[("enhanced", index) for index in sorted(item_geometry_refs.get(id(item), set()))],
        )

    for item in active_items:
        current_area_sqm = item.area_value or 0.0
        if current_area_sqm >= SHARED_FRAGMENT_FALLBACK_MAX_EXISTING_AREA_SQM:
            continue

        point = Point(item.cx, item.cy)
        containing_indices = sorted(
            [index for index in deferred_indices if polygons[index].buffer(0.5).contains(point)],
            key=lambda index: polygon_areas_sqm[index],
        )
        if len(containing_indices) != 1:
            continue

        fallback_index = containing_indices[0]
        fallback_area_sqm = polygon_areas_sqm[fallback_index]
        if fallback_area_sqm < OPEN_FRAGMENT_MIN_ASSIGN_AREA_SQM:
            continue

        fill_ratio = _polygon_fill_ratio(polygons[fallback_index])
        if fallback_area_sqm > AMBIGUOUS_FRAGMENT_AREA_SQM and fill_ratio < AMBIGUOUS_FRAGMENT_FILL_RATIO:
            fallback_area_sqm *= SHARED_FRAGMENT_FALLBACK_FACTOR

        _assign_exact_area(
            item,
            fallback_area_sqm,
            "vector_shared_fragment_fallback",
            0.28,
            geometry_refs=[("enhanced", fallback_index)],
        )


def _assign_residual_open_area(
    all_items: List[TextItem],
    unresolved_items: List[TextItem],
    enhanced_polygons: List[Polygon],
    enhanced_polygon_areas_sqm: List[float],
    enhanced_polygon_tree: STRtree | None = None,
) -> None:
    parent_groups: Dict[int, List[TextItem]] = defaultdict(list)
    nearest_fragment_distance: Dict[int, float] = {}

    fragment_indices = _build_fragment_indices(enhanced_polygon_areas_sqm)

    for item in unresolved_items:
        if item.area_value is not None:
            continue
        point = Point(item.cx, item.cy)
        parent_index = _find_parent_polygon_index(
            point,
            enhanced_polygons,
            enhanced_polygon_areas_sqm,
            enhanced_polygon_tree,
        )
        if parent_index is None:
            continue
        parent_groups[parent_index].append(item)

        nearest_fragment_distance[id(item)] = _nearest_fragment_distance(point, enhanced_polygons, fragment_indices)

    for parent_index, group_items in parent_groups.items():
        parent_area_sqm = enhanced_polygon_areas_sqm[parent_index]
        parent_polygon = enhanced_polygons[parent_index]
        already_claimed_sqm = 0.0
        for item in all_items:
            if item.area_value is None:
                continue
            if parent_polygon.buffer(0.5).contains(Point(item.cx, item.cy)):
                already_claimed_sqm += item.area_value or 0.0
        residual_area_sqm = max(parent_area_sqm - already_claimed_sqm, 0.0)
        if residual_area_sqm < OPEN_RESIDUAL_MIN_AREA_SQM:
            continue

        if len(group_items) >= 2:
            weights = _estimate_residual_partition_weights(
                parent_polygon=parent_polygon,
                group_items=group_items,
                nearest_fragment_distance=nearest_fragment_distance,
            )
            if not weights:
                continue

            for item in group_items:
                share = max(weights.get(id(item), 0.0), 0.0)
                if share <= 0:
                    continue
                item.area_value = round(residual_area_sqm * share, 2)
                item.area_method = "vector_open_residual_partition"
                item.area_confidence = 0.24
                item.area_geometry_refs = [("enhanced", parent_index)]
            continue

        ranked_items = sorted(
            group_items,
            key=lambda item: nearest_fragment_distance.get(id(item), 0.0),
            reverse=True,
        )
        top_distance = nearest_fragment_distance.get(id(ranked_items[0]), 0.0)
        next_distance = nearest_fragment_distance.get(id(ranked_items[1]), 0.0) if len(ranked_items) > 1 else 0.0
        if top_distance < OPEN_RESIDUAL_MIN_DISTANCE_PDF:
            continue
        if next_distance > 0 and top_distance < (next_distance * OPEN_RESIDUAL_DOMINANCE_RATIO):
            continue

        winner = ranked_items[0]
        winner.area_value = round((winner.area_value or 0.0) + residual_area_sqm, 2)
        winner.area_method = "vector_open_residual"
        winner.area_confidence = 0.33
        winner.area_geometry_refs = [("enhanced", parent_index)]


def _estimate_residual_partition_weights(
    parent_polygon: Polygon,
    group_items: List[TextItem],
    nearest_fragment_distance: Dict[int, float],
) -> Dict[int, float]:
    if not group_items:
        return {}

    item_points = {id(item): Point(item.cx, item.cy) for item in group_items}
    counts: Dict[int, int] = defaultdict(int)
    total_samples = 0

    min_x, min_y, max_x, max_y = parent_polygon.bounds
    step = max(RESIDUAL_PARTITION_GRID_STEP_PDF, 1.0)

    y = min_y
    while y <= max_y:
        x = min_x
        while x <= max_x:
            sample_point = Point(x, y)
            if parent_polygon.buffer(0.1).contains(sample_point):
                winner = min(group_items, key=lambda item: sample_point.distance(item_points[id(item)]))
                counts[id(winner)] += 1
                total_samples += 1
            x += step
        y += step

    if total_samples == 0:
        return {id(item): 1.0 / len(group_items) for item in group_items}

    scores: Dict[int, float] = {}
    for item in group_items:
        local_share = counts.get(id(item), 0) / total_samples
        local_term = max(local_share, 1e-6) ** RESIDUAL_PARTITION_LOCAL_SHARE_POWER
        font_term = max(item.font_size, 1.0) ** RESIDUAL_PARTITION_FONT_POWER
        fragment_distance = min(
            nearest_fragment_distance.get(id(item), 0.0),
            RESIDUAL_PARTITION_MAX_DISTANCE_PDF,
        )
        fragment_term = exp(fragment_distance / RESIDUAL_PARTITION_FRAGMENT_SCALE_PDF)
        scores[id(item)] = local_term * font_term * fragment_term

    total_score = sum(scores.values())
    if total_score <= 0:
        return {id(item): 1.0 / len(group_items) for item in group_items}

    return {item_id: score / total_score for item_id, score in scores.items()}


def estimate_page_label_areas(
    pdf_path: str,
    page_number: int,
    label_candidates: List[TextItem],
) -> Dict[str, Any]:
    scale_ratio = extract_page_scale_ratio(pdf_path, page_number)
    polygons = build_page_vector_polygons(pdf_path, page_number)
    polygon_tree = STRtree(polygons) if polygons else None
    if scale_ratio is None and polygons:
        scale_ratio = _infer_scale_ratio_from_embedded_areas(label_candidates, polygons, polygon_tree)
    if scale_ratio is None and polygons:
        scale_ratio = _infer_scale_ratio_from_size_annotations(
            extract_pdf_lines(pdf_path, page_number=page_number),
            polygons,
        )
    polygon_areas_sqm = [_area_pdf_to_sqm(polygon.area, scale_ratio) for polygon in polygons] if scale_ratio else []

    resolved_count = 0
    unresolved_count = 0

    if scale_ratio is None or not polygons:
        for item in label_candidates:
            if item.embedded_area_value is not None:
                _assign_text_embedded_area(item)
                resolved_count += 1
            else:
                _assign_unresolved(item)
                unresolved_count += 1

        return {
            "scale_ratio": scale_ratio,
            "polygon_count": len(polygons),
            "resolved_area_count": resolved_count,
            "unresolved_area_count": unresolved_count,
        }

    for item in label_candidates:
        point = Point(item.cx, item.cy)
        containing_indices = _find_containing_polygon_indices(point, polygons, polygon_tree)

        if not containing_indices:
            if item.embedded_area_value is not None:
                _assign_text_embedded_area(item)
            else:
                _assign_unresolved(item)
            continue

        area_sqm = polygon_areas_sqm[containing_indices[0]]
        if area_sqm <= MAX_EXACT_VECTOR_AREA_SQM:
            _assign_exact_area(
                item,
                area_sqm,
                "vector_polygon_exact",
                0.95,
                geometry_refs=[("base", containing_indices[0])],
            )
        else:
            if item.embedded_area_value is not None:
                _assign_text_embedded_area(item)
            else:
                _assign_unresolved(item)

    high_complexity_page = len(polygons) > 2000 and len(label_candidates) > 150
    if high_complexity_page:
        resolved_count = sum(1 for item in label_candidates if item.area_value is not None)
        unresolved_count = len(label_candidates) - resolved_count
        return {
            "scale_ratio": scale_ratio,
            "polygon_count": len(polygons),
            "resolved_area_count": resolved_count,
            "unresolved_area_count": unresolved_count,
        }

    door_aware_polygons = build_page_vector_polygons(
        pdf_path,
        page_number,
        close_door_arcs=True,
        use_door_gap_bridges=True,
    )
    door_aware_polygon_tree = STRtree(door_aware_polygons) if door_aware_polygons else None
    door_aware_polygon_areas_sqm = (
        [_area_pdf_to_sqm(polygon.area, scale_ratio) for polygon in door_aware_polygons] if scale_ratio else []
    )

    if door_aware_polygons and door_aware_polygon_areas_sqm:
        _assign_direct_second_pass_areas(
            label_candidates,
            door_aware_polygons,
            door_aware_polygon_areas_sqm,
            door_aware_polygon_tree,
        )
        _reopen_networked_closed_gap_labels(
            label_candidates,
            door_aware_polygons,
            door_aware_polygon_areas_sqm,
            door_aware_polygon_tree,
        )

    enhanced_polygons = build_page_vector_polygons(pdf_path, page_number, close_door_arcs=True)
    enhanced_polygon_tree = STRtree(enhanced_polygons) if enhanced_polygons else None
    enhanced_polygon_areas_sqm = (
        [_area_pdf_to_sqm(polygon.area, scale_ratio) for polygon in enhanced_polygons] if scale_ratio else []
    )

    unresolved_items = [item for item in label_candidates if item.area_value is None]
    resolved_items = [item for item in label_candidates if item.area_value is not None]
    if unresolved_items and enhanced_polygons:
        _assign_fragment_partition_areas(
            unresolved_items,
            resolved_items,
            enhanced_polygons,
            enhanced_polygon_areas_sqm,
        )

    unresolved_items = [item for item in label_candidates if item.area_value is None]
    if unresolved_items and enhanced_polygons:
        _assign_sparse_isolated_fragment_areas(
            unresolved_items,
            label_candidates,
            enhanced_polygons,
            enhanced_polygon_areas_sqm,
            enhanced_polygon_tree,
        )

    unresolved_items = [item for item in label_candidates if item.area_value is None]
    if unresolved_items and enhanced_polygons:
        _assign_residual_open_area(
            label_candidates,
            unresolved_items,
            enhanced_polygons,
            enhanced_polygon_areas_sqm,
            enhanced_polygon_tree,
        )

    for item in label_candidates:
        if item.area_value is None and item.embedded_area_value is not None:
            _assign_text_embedded_area(item)

    resolved_count = sum(1 for item in label_candidates if item.area_value is not None)
    unresolved_count = len(label_candidates) - resolved_count

    return {
        "scale_ratio": scale_ratio,
        "polygon_count": len(enhanced_polygons),
        "resolved_area_count": resolved_count,
        "unresolved_area_count": unresolved_count,
    }
