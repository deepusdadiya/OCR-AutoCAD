import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import fitz
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from src.area_calculation import build_page_vector_polygons
from src.models import TextItem


BOUNDARY_COLOR = (0, 0, 0)
CENTER_MARKER_COLOR = (220, 20, 60)
BOUNDARY_WIDTH = 4
CROP_PADDING_PX = 32
CAPTION_BAND_HEIGHT = 34


def build_area_boundaries_visualization_path(pdf_path: str | Path, output_dir: Path) -> Path:
    stem = Path(pdf_path).stem
    normalized_stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower() or "document"
    return output_dir / f"area_boundaries_visualization_{normalized_stem}.pdf"


def _page_polygon_sets(pdf_path: str | Path, page_number: int) -> Dict[str, List[Polygon]]:
    return {
        "base": build_page_vector_polygons(str(pdf_path), page_number),
        "door_aware": build_page_vector_polygons(
            str(pdf_path),
            page_number,
            close_door_arcs=True,
            use_door_gap_bridges=True,
        ),
        "enhanced": build_page_vector_polygons(str(pdf_path), page_number, close_door_arcs=True),
    }


def _page_rotation_matrix(pdf_path: str | Path, page_number: int) -> fitz.Matrix:
    doc = fitz.open(str(pdf_path))
    page = doc[page_number]
    rotation_matrix = fitz.Matrix(page.rotation_matrix)
    doc.close()
    return rotation_matrix


def _scale_point(
    point: Tuple[float, float],
    rotation_matrix: fitz.Matrix,
    page_size: Tuple[float, float],
    image_size: Tuple[int, int],
) -> Tuple[float, float]:
    page_w, page_h = page_size
    image_w, image_h = image_size
    if page_w <= 0 or page_h <= 0:
        return point

    rotated_point = fitz.Point(float(point[0]), float(point[1])) * rotation_matrix
    return (
        float(rotated_point.x) * image_w / page_w,
        float(rotated_point.y) * image_h / page_h,
    )


def _draw_polygon_outline(
    draw: ImageDraw.ImageDraw,
    polygon: Polygon,
    rotation_matrix: fitz.Matrix,
    page_size: Tuple[float, float],
    image_size: Tuple[int, int],
) -> None:
    coords = list(polygon.exterior.coords)
    if len(coords) < 2:
        return

    scaled = [_scale_point((x, y), rotation_matrix, page_size, image_size) for x, y in coords]
    draw.line(scaled, fill=BOUNDARY_COLOR, width=BOUNDARY_WIDTH)


def _draw_center_marker(
    draw: ImageDraw.ImageDraw,
    item: TextItem,
    rotation_matrix: fitz.Matrix,
    page_size: Tuple[float, float],
    image_size: Tuple[int, int],
) -> None:
    cx, cy = _scale_point((item.cx, item.cy), rotation_matrix, page_size, image_size)
    radius = 5
    draw.ellipse(
        [(cx - radius, cy - radius), (cx + radius, cy + radius)],
        fill=CENTER_MARKER_COLOR,
        outline=CENTER_MARKER_COLOR,
    )


def _resolve_item_polygons(
    item: TextItem,
    polygon_sets: Dict[str, List[Polygon]],
) -> List[Polygon]:
    if item.area_geometry_shapes:
        polygons: List[Polygon] = []
        for geometry in item.area_geometry_shapes:
            if geometry is None or geometry.is_empty:
                continue
            if geometry.geom_type == "Polygon":
                polygons.append(geometry)
            else:
                polygons.extend(
                    part
                    for part in getattr(geometry, "geoms", [])
                    if getattr(part, "geom_type", "") == "Polygon"
                )
        if polygons:
            return polygons

    polygons: List[Polygon] = []
    seen: set[Tuple[str, int]] = set()

    for ref_name, polygon_index in item.area_geometry_refs:
        ref_key = (ref_name, polygon_index)
        if ref_key in seen:
            continue
        seen.add(ref_key)

        source_polygons = polygon_sets.get(ref_name)
        if source_polygons is None:
            continue
        if polygon_index < 0 or polygon_index >= len(source_polygons):
            continue
        polygons.append(source_polygons[polygon_index])

    return polygons


def _collect_geometry_bounds(
    item: TextItem,
    polygons: Sequence[Polygon],
) -> Tuple[float, float, float, float]:
    bounds = [item.bbox]
    bounds.extend(polygon.bounds for polygon in polygons)
    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)
    return (min_x, min_y, max_x, max_y)


def _crop_image_for_geometry(
    image: Image.Image,
    image_points: Sequence[Tuple[float, float]],
) -> Tuple[Image.Image, Tuple[int, int]]:
    image_w, image_h = image.size
    min_x = min(point[0] for point in image_points)
    min_y = min(point[1] for point in image_points)
    max_x = max(point[0] for point in image_points)
    max_y = max(point[1] for point in image_points)

    crop_box = (
        max(int(min_x) - CROP_PADDING_PX, 0),
        max(int(min_y) - CROP_PADDING_PX, 0),
        min(int(max_x) + CROP_PADDING_PX, image_w),
        min(int(max_y) + CROP_PADDING_PX, image_h),
    )
    return image.crop(crop_box), (crop_box[0], crop_box[1])


def _add_caption_band(image: Image.Image, caption: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + CAPTION_BAND_HEIGHT), "white")
    canvas.paste(image, (0, CAPTION_BAND_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), caption, fill="black")
    return canvas


def _bbox_image_points(
    bbox: Tuple[float, float, float, float],
    rotation_matrix: fitz.Matrix,
    page_size: Tuple[float, float],
    image_size: Tuple[int, int],
) -> List[Tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [_scale_point(point, rotation_matrix, page_size, image_size) for point in corners]


def _polygon_image_points(
    polygon: Polygon,
    rotation_matrix: fitz.Matrix,
    page_size: Tuple[float, float],
    image_size: Tuple[int, int],
) -> List[Tuple[float, float]]:
    return [
        _scale_point((x, y), rotation_matrix, page_size, image_size)
        for x, y in polygon.exterior.coords
    ]


def _collect_geometry_image_points(
    item: TextItem,
    polygons: Sequence[Polygon],
    rotation_matrix: fitz.Matrix,
    page_size: Tuple[float, float],
    image_size: Tuple[int, int],
) -> List[Tuple[float, float]]:
    image_points = _bbox_image_points(item.bbox, rotation_matrix, page_size, image_size)
    for polygon in polygons:
        image_points.extend(_polygon_image_points(polygon, rotation_matrix, page_size, image_size))
    return image_points


def _render_overview_page(
    page_image: Image.Image,
    rotation_matrix: fitz.Matrix,
    page_size: Tuple[float, float],
    polygons: Iterable[Polygon],
    page_number: int,
) -> Image.Image:
    image = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        _draw_polygon_outline(draw, polygon, rotation_matrix, page_size, image.size)
    return _add_caption_band(image, f"Page {page_number + 1} - final area boundaries")


def _render_item_crop_page(
    page_image: Image.Image,
    rotation_matrix: fitz.Matrix,
    page_size: Tuple[float, float],
    item: TextItem,
    polygons: Sequence[Polygon],
) -> Image.Image:
    image = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        _draw_polygon_outline(draw, polygon, rotation_matrix, page_size, image.size)
    _draw_center_marker(draw, item, rotation_matrix, page_size, image.size)

    geometry_image_points = _collect_geometry_image_points(item, polygons, rotation_matrix, page_size, image.size)
    cropped, offset = _crop_image_for_geometry(image, geometry_image_points)

    marker_image = cropped.copy()
    marker_draw = ImageDraw.Draw(marker_image)
    local_center_x, local_center_y = _scale_point((item.cx, item.cy), rotation_matrix, page_size, image.size)
    local_center_x -= offset[0]
    local_center_y -= offset[1]
    radius = 5
    marker_draw.ellipse(
        [
            (local_center_x - radius, local_center_y - radius),
            (local_center_x + radius, local_center_y + radius),
        ],
        fill=CENTER_MARKER_COLOR,
        outline=CENTER_MARKER_COLOR,
    )
    caption = f"{item.text} | {item.area_method} | {item.area_value if item.area_value is not None else 'n/a'}"
    return _add_caption_band(marker_image, caption)


def save_area_boundaries_visualization_pdf(
    pdf_path: str | Path,
    page_results: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    if not page_results:
        return None

    output_path = build_area_boundaries_visualization_path(pdf_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_pages: List[Image.Image] = []

    for page_result in page_results:
        page_number = page_result["page_number"]
        page_image = page_result["full_image"]
        page_size = page_result["page_size"]
        rotation_matrix = _page_rotation_matrix(pdf_path, page_number)
        polygon_sets = _page_polygon_sets(pdf_path, page_number)
        items = [
            item
            for item in page_result["fused_label_candidates"]
            if item.area_geometry_refs
        ]
        if not items:
            continue

        overview_refs = {
            ref
            for item in items
            for ref in item.area_geometry_refs
        }
        overview_polygons = [
            polygon_sets[ref_name][index]
            for ref_name, index in sorted(overview_refs)
            if ref_name in polygon_sets and 0 <= index < len(polygon_sets[ref_name])
        ]
        rendered_pages.append(
            _render_overview_page(
                page_image=page_image,
                rotation_matrix=rotation_matrix,
                page_size=page_size,
                polygons=overview_polygons,
                page_number=page_number,
            )
        )

        for item in items:
            item_polygons = _resolve_item_polygons(item, polygon_sets)
            if not item_polygons:
                continue
            rendered_pages.append(
                _render_item_crop_page(
                    page_image=page_image,
                    rotation_matrix=rotation_matrix,
                    page_size=page_size,
                    item=item,
                    polygons=item_polygons,
                )
            )

    if not rendered_pages:
        return None

    first_page, *remaining_pages = rendered_pages
    first_page.save(output_path, save_all=True, append_images=remaining_pages)
    return output_path
