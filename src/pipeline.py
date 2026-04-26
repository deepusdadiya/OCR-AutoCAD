from pathlib import Path
from typing import Dict, Any

from config import (
    RENDER_DPI,
    MIN_DRAWING_COMPONENT_AREA,
    DRAWING_REGION_PADDING,
)
from src.area_calculation import estimate_page_label_areas
from src.evaluate import compare_with_expected
from src.pdf_io import get_page_count, get_page_size, render_pdf_page
from src.label_fusion import fuse_label_candidates
from src.text_extraction import extract_text_payload
from src.page_analysis import detect_page_regions, draw_page_regions
from src.text_block_reconstruction import reconstruct_text_blocks, deduplicate_blocks
from src.text_classifier import assign_region_type, classify_text_items, keep_room_label_candidates
from src.output_builder import build_text_candidates_df
from src.instance_export import (
    build_final_client_instances_df,
    build_final_client_instances_debug_df,
)


def _run_page_pipeline(pdf_path: str, page_number: int) -> Dict[str, Any]:
    page_w, page_h = get_page_size(pdf_path, page_number=page_number)
    full_image = render_pdf_page(pdf_path, page_number=page_number, dpi=RENDER_DPI)

    page_regions = detect_page_regions(
        full_image,
        min_component_area=MIN_DRAWING_COMPONENT_AREA,
        drawing_region_padding=DRAWING_REGION_PADDING,
    )
    page_region_debug = draw_page_regions(full_image, page_regions)

    extraction_payload = extract_text_payload(pdf_path, page_number=page_number)
    raw_words = extraction_payload["items"]
    reconstructed_blocks = reconstruct_text_blocks(raw_words)
    reconstructed_blocks = deduplicate_blocks(reconstructed_blocks)

    reconstructed_blocks = assign_region_type(
        reconstructed_blocks,
        page_regions=page_regions,
        page_w=page_w,
        page_h=page_h,
        img_w=full_image.size[0],
        img_h=full_image.size[1],
    )

    classified_items = classify_text_items(reconstructed_blocks)
    label_candidates = keep_room_label_candidates(classified_items)
    fused_label_candidates = fuse_label_candidates(label_candidates)
    area_meta = estimate_page_label_areas(pdf_path, page_number, fused_label_candidates)

    return {
        "page_number": page_number,
        "page_size": (page_w, page_h),
        "full_image": full_image,
        "page_region_debug": page_region_debug,
        "page_regions": page_regions,
        "text_extraction_mode": extraction_payload["mode"],
        "ocr_available": extraction_payload["ocr_available"],
        "ocr_attempted": extraction_payload["ocr_attempted"],
        "vector_alpha_count": extraction_payload["vector_alpha_count"],
        "ocr_alpha_count": extraction_payload["ocr_alpha_count"],
        "ocr_rotation_hits": extraction_payload["ocr_rotation_hits"],
        "scale_ratio": area_meta["scale_ratio"],
        "area_polygon_count": area_meta["polygon_count"],
        "resolved_area_count": area_meta["resolved_area_count"],
        "unresolved_area_count": area_meta["unresolved_area_count"],
        "raw_words": raw_words,
        "reconstructed_blocks": reconstructed_blocks,
        "classified_items": classified_items,
        "label_candidates": label_candidates,
        "fused_label_candidates": fused_label_candidates,
        "text_candidate_count": len(classified_items),
        "final_label_count": len(fused_label_candidates),
    }


def run_pipeline(pdf_path: str, expected_csv_path: str | None = None) -> Dict[str, Any]:
    page_count = get_page_count(pdf_path)
    page_results = [_run_page_pipeline(pdf_path, page_number) for page_number in range(page_count)]

    raw_words = [item for page in page_results for item in page["raw_words"]]
    reconstructed_blocks = [item for page in page_results for item in page["reconstructed_blocks"]]
    classified_items = [item for page in page_results for item in page["classified_items"]]
    label_candidates = [item for page in page_results for item in page["label_candidates"]]
    fused_label_candidates = [item for page in page_results for item in page["fused_label_candidates"]]

    text_df = build_text_candidates_df(classified_items)
    final_client_df = build_final_client_instances_df(fused_label_candidates)
    final_client_debug_df = build_final_client_instances_debug_df(fused_label_candidates)

    comparison_df = None
    if expected_csv_path and Path(expected_csv_path).exists():
        comparison_df = compare_with_expected(final_client_df, expected_csv_path)

    first_page = page_results[0] if page_results else None

    return {
        "page_count": page_count,
        "page_results": page_results,
        "full_image": first_page["full_image"] if first_page else None,
        "page_region_debug": first_page["page_region_debug"] if first_page else None,
        "page_regions": first_page["page_regions"] if first_page else None,
        "raw_words": raw_words,
        "reconstructed_blocks": reconstructed_blocks,
        "classified_items": classified_items,
        "label_candidates": label_candidates,
        "text_df": text_df,
        "final_client_df": final_client_df,
        "final_client_debug_df": final_client_debug_df,
        "fused_label_candidates": fused_label_candidates,
        "comparison_df": comparison_df,
    }
