from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import streamlit as st

from config import INPUT_PDF
from src.pipeline import run_pipeline
from src.runtime_inputs import resolve_expected_csv_path


st.set_page_config(page_title="Generic Floor Plan OCR POC", layout="wide")
st.title("Generic Floor Plan OCR POC")

if "pipeline_result" not in st.session_state:
    st.session_state["pipeline_result"] = None
    st.session_state["pipeline_source"] = ""
    st.session_state["pipeline_expected"] = ""


def _write_uploaded_file(uploaded_file: Any, directory: Path) -> Path:
    target_path = directory / Path(uploaded_file.name).name
    target_path.write_bytes(uploaded_file.getvalue())
    return target_path


def _run_selected_pipeline(
    pdf_upload: Any | None,
    expected_upload: Any | None,
) -> tuple[dict, str, str]:
    if pdf_upload is None:
        pdf_path = INPUT_PDF
        expected_path = resolve_expected_csv_path(pdf_path, None)
        result = run_pipeline(str(pdf_path), str(expected_path) if expected_path else None)
        expected_label = expected_path.name if expected_path is not None else ""
        return result, pdf_path.name, expected_label

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = _write_uploaded_file(pdf_upload, temp_path)
        expected_path = _write_uploaded_file(expected_upload, temp_path) if expected_upload else None
        result = run_pipeline(str(pdf_path), str(expected_path) if expected_path else None)
        expected_label = expected_upload.name if expected_upload else ""
        return result, pdf_upload.name, expected_label


st.caption(
    f"Upload a PDF to process a new drawing, or leave it empty to run the bundled sample `{INPUT_PDF.name}`."
)
uploaded_pdf = st.file_uploader("Input PDF", type=["pdf"])
uploaded_expected = st.file_uploader("Expected CSV (optional)", type=["csv"])

if st.button("Run Pipeline"):
    with st.spinner("Processing..."):
        result, source_label, expected_label = _run_selected_pipeline(uploaded_pdf, uploaded_expected)
    st.session_state["pipeline_result"] = result
    st.session_state["pipeline_source"] = source_label
    st.session_state["pipeline_expected"] = expected_label

result = st.session_state["pipeline_result"]
if result is None:
    st.info("Upload a PDF or use the sample, then click Run Pipeline.")
else:
    source_label = st.session_state["pipeline_source"]
    expected_label = st.session_state["pipeline_expected"]
    st.caption(f"Current source: `{source_label}`")
    if expected_label:
        st.caption(f"Comparison CSV: `{expected_label}`")

    selected_page = st.selectbox(
        "Page",
        options=list(range(result["page_count"])),
        format_func=lambda page_number: f"Page {page_number + 1}",
    )
    page_result = result["page_results"][selected_page]
    page_text_df = result["text_df"][result["text_df"]["page_number"] == selected_page]
    page_final_labels_df = result["final_client_debug_df"][
        result["final_client_debug_df"]["page_number"] == selected_page
    ]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Page Regions", "Text Candidates", "Final Labels", "Expected vs Predicted"]
    )

    with tab1:
        st.caption(
            f"Processed {result['page_count']} page(s). "
            f"This page has {page_result['text_candidate_count']} text candidates and "
            f"{page_result['final_label_count']} final labels. "
            f"Resolved areas: {page_result['resolved_area_count']}/{page_result['final_label_count']}. "
            f"Extraction mode: {page_result['text_extraction_mode']}."
        )
        if page_result["scale_ratio"] is not None:
            st.caption(
                f"Detected scale 1:{page_result['scale_ratio']:.0f}. "
                f"Vector polygons: {page_result['area_polygon_count']}."
            )
        if page_result["ocr_attempted"]:
            st.caption(
                f"OCR available: {page_result['ocr_available']}. "
                f"Vector alpha items: {page_result['vector_alpha_count']}. "
                f"OCR alpha items: {page_result['ocr_alpha_count']}."
            )
        st.image(page_result["full_image"], caption="Rendered page", width="stretch")
        st.image(page_result["page_region_debug"], caption="Detected drawing/metadata regions", width="stretch")

    with tab2:
        st.dataframe(page_text_df, width="stretch")

    with tab3:
        st.dataframe(page_final_labels_df, width="stretch")

    with tab4:
        if result["comparison_df"] is not None:
            st.dataframe(result["comparison_df"], width="stretch")
            matched = int(result["comparison_df"]["matched"].sum())
            total = len(result["comparison_df"])
            st.metric("Matched expected names", f"{matched}/{total}")
        else:
            st.info("No expected CSV provided for this run.")
