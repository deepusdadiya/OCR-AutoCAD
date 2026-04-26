import streamlit as st

from config import INPUT_PDF, EXPECTED_CSV
from src.pipeline import run_pipeline


st.set_page_config(page_title="Generic Floor Plan OCR POC", layout="wide")
st.title("Generic Floor Plan OCR POC")

if st.button("Run Pipeline"):
    with st.spinner("Processing..."):
        result = run_pipeline(str(INPUT_PDF), str(EXPECTED_CSV))

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
        st.image(page_result["full_image"], caption="Rendered page", use_container_width=True)
        st.image(page_result["page_region_debug"], caption="Detected drawing/metadata regions", use_container_width=True)

    with tab2:
        st.dataframe(page_text_df, use_container_width=True)

    with tab3:
        st.dataframe(page_final_labels_df, use_container_width=True)

    with tab4:
        if result["comparison_df"] is not None:
            st.dataframe(result["comparison_df"], use_container_width=True)
            matched = int(result["comparison_df"]["matched"].sum())
            total = len(result["comparison_df"])
            st.metric("Matched expected names", f"{matched}/{total}")
        else:
            st.info("No expected CSV found.")
else:
    st.info("Click the button to run the pipeline.")
