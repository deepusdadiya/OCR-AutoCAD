from typing import Dict, Any

from config import (
    RENDER_DPI,
    MIN_DRAWING_COMPONENT_AREA,
    DRAWING_REGION_PADDING,
    WORD_Y_TOL,
    WORD_X_GAP_TOL,
    MIN_ROOM_AREA_PX,
    MAX_ROOM_AREA_RATIO,
    WALL_BINARY_THRESHOLD,
    MORPH_CLOSE_KERNEL,
    MORPH_CLOSE_ITER,
)
from src.pdf_io import get_page_size, render_pdf_page
from src.text_extraction import extract_text_items
from src.page_analysis import detect_page_regions, crop_to_bbox, draw_page_regions
from src.text_grouping import group_words_into_phrases, deduplicate_phrases
from src.text_classifier import assign_region_type, classify_text_items, keep_room_label_candidates
from src.geometry_extraction import detect_room_candidates, draw_room_candidates
from src.assignment import assign_labels_to_rooms
from src.output_builder import (
    build_text_candidates_df,
    build_rooms_df,
    build_label_room_matches_df,
    build_final_output_df,
    build_final_debug_df,
)
from src.evaluate import compare_with_expected


def run_pipeline(pdf_path: str, expected_csv_path: str | None = None) -> Dict[str, Any]:
    page_w, page_h = get_page_size(pdf_path, page_number=0)
    full_image = render_pdf_page(pdf_path, page_number=0, dpi=RENDER_DPI)

    page_regions = detect_page_regions(
        full_image,
        min_component_area=MIN_DRAWING_COMPONENT_AREA,
        drawing_region_padding=DRAWING_REGION_PADDING,
    )
    page_region_debug = draw_page_regions(full_image, page_regions)

    drawing_crop, crop_offset = crop_to_bbox(full_image, page_regions.drawing_bbox_img)

    raw_words = extract_text_items(pdf_path, page_number=0)
    grouped_phrases = group_words_into_phrases(raw_words, y_tol=WORD_Y_TOL, x_gap_tol=WORD_X_GAP_TOL)
    grouped_phrases = deduplicate_phrases(grouped_phrases)

    grouped_phrases = assign_region_type(
        grouped_phrases,
        page_regions=page_regions,
        page_w=page_w,
        page_h=page_h,
        img_w=full_image.size[0],
        img_h=full_image.size[1],
    )
    classified_items = classify_text_items(grouped_phrases)
    label_candidates = keep_room_label_candidates(classified_items)

    room_candidates = detect_room_candidates(
        drawing_crop,
        min_room_area_px=MIN_ROOM_AREA_PX,
        max_room_area_ratio=MAX_ROOM_AREA_RATIO,
        wall_binary_threshold=WALL_BINARY_THRESHOLD,
        morph_close_kernel=MORPH_CLOSE_KERNEL,
        morph_close_iter=MORPH_CLOSE_ITER,
    )

    assigned_rooms = assign_labels_to_rooms(
        rooms=room_candidates,
        text_items=label_candidates,
        page_w=page_w,
        page_h=page_h,
        img_w=full_image.size[0],
        img_h=full_image.size[1],
        crop_offset=crop_offset,
    )

    room_debug = draw_room_candidates(drawing_crop, assigned_rooms)

    text_df = build_text_candidates_df(classified_items)
    rooms_df = build_rooms_df(assigned_rooms)
    label_matches_df = build_label_room_matches_df(
        label_candidates=label_candidates,
        rooms=assigned_rooms,
        page_w=page_w,
        page_h=page_h,
        img_w=full_image.size[0],
        img_h=full_image.size[1],
        crop_offset=crop_offset,
    )
    final_df = build_final_output_df(label_matches_df)
    final_debug_df = build_final_debug_df(label_matches_df)
    comparison_df = None
    if expected_csv_path:
        comparison_df = compare_with_expected(rooms_df.copy(), expected_csv_path)

    return {
        "full_image": full_image,
        "drawing_crop": drawing_crop,
        "page_region_debug": page_region_debug,
        "room_debug": room_debug,
        "page_regions": page_regions,
        "raw_words": raw_words,
        "grouped_phrases": grouped_phrases,
        "classified_items": classified_items,
        "label_candidates": label_candidates,
        "room_candidates": assigned_rooms,
        "text_df": text_df,
        "rooms_df": rooms_df,
        "label_matches_df": label_matches_df,
        "comparison_df": comparison_df,
        "crop_offset": crop_offset,
        "final_df": final_df,
        "final_debug_df": final_debug_df,
    }