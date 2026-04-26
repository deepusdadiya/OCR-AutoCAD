import streamlit as st

from config import INPUT_PDF, EXPECTED_CSV
from src.pipeline import run_pipeline


st.set_page_config(page_title="Generic Floor Plan OCR POC", layout="wide")
st.title("Generic Floor Plan OCR POC")

if st.button("Run Pipeline"):
    with st.spinner("Processing..."):
        result = run_pipeline(str(INPUT_PDF), str(EXPECTED_CSV))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Page Regions", "Text Candidates", "Final Labels", "Expected vs Predicted"]
    )

    with tab1:
        st.image(result["full_image"], caption="Rendered page", use_container_width=True)
        st.image(result["page_region_debug"], caption="Detected drawing/metadata regions", use_container_width=True)

    with tab2:
        st.dataframe(result["text_df"], use_container_width=True)

    with tab3:
        st.dataframe(result["final_client_debug_df"], use_container_width=True)

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