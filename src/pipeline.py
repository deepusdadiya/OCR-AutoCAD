from typing import Dict, Any

from config import (
    RENDER_DPI,
    MIN_DRAWING_COMPONENT_AREA,
    DRAWING_REGION_PADDING,
)
from src.pdf_io import get_page_size, render_pdf_page
from src.label_fusion import fuse_label_candidates
from src.text_extraction import extract_text_items
from src.page_analysis import detect_page_regions, draw_page_regions
from src.text_block_reconstruction import reconstruct_text_blocks, deduplicate_blocks
from src.text_classifier import assign_region_type, classify_text_items, keep_room_label_candidates
from src.output_builder import build_text_candidates_df
from src.instance_export import (
    build_final_client_instances_df,
    build_final_client_instances_debug_df,
)


def run_pipeline(pdf_path: str, expected_csv_path: str | None = None) -> Dict[str, Any]:
    page_w, page_h = get_page_size(pdf_path, page_number=0)
    full_image = render_pdf_page(pdf_path, page_number=0, dpi=RENDER_DPI)

    page_regions = detect_page_regions(
        full_image,
        min_component_area=MIN_DRAWING_COMPONENT_AREA,
        drawing_region_padding=DRAWING_REGION_PADDING,
    )
    page_region_debug = draw_page_regions(full_image, page_regions)

    raw_words = extract_text_items(pdf_path, page_number=0)

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

    text_df = build_text_candidates_df(classified_items)
    final_client_df = build_final_client_instances_df(fused_label_candidates)
    final_client_debug_df = build_final_client_instances_debug_df(fused_label_candidates)

    return {
        "full_image": full_image,
        "page_region_debug": page_region_debug,
        "page_regions": page_regions,
        "raw_words": raw_words,
        "reconstructed_blocks": reconstructed_blocks,
        "classified_items": classified_items,
        "label_candidates": label_candidates,
        "text_df": text_df,
        "final_client_df": final_client_df,
        "final_client_debug_df": final_client_debug_df,
        "fused_label_candidates": fused_label_candidates,
    }