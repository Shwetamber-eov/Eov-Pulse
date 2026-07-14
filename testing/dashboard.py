import streamlit as st
import pandas as pd

from runner import run_all_tests

st.set_page_config(
    page_title="Clinical AI Test Dashboard",
    layout="wide"
)

st.title("🩺 Clinical AI test Dashboard")

st.write(
    "Runs all PDFs inside **test_documents/** through the extraction "
    "and clinical analysis pipeline."
)

if st.button("▶ Run All Tests", use_container_width=True):

    with st.spinner("Running tests..."):

        results = run_all_tests()

    passed = sum(r.status == "PASS" for r in results)
    failed = len(results) - passed

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Reports", len(results))
    c2.metric("Passed", passed)
    c3.metric("Failed", failed)

    st.divider()

    summary = pd.DataFrame([
        {
            "Report": r.filename,
            "Status": r.status,
            "Time (sec)": r.execution_time
        }
        for r in results
    ])

    st.dataframe(summary, use_container_width=True)

    st.divider()

    for result in results:

        icon = "✅" if result.status == "PASS" else "❌"

        with st.expander(f"{icon} {result.filename}"):

            st.write(f"**Status:** {result.status}")
            st.write(f"**Execution Time:** {result.execution_time} sec")

            if result.status == "FAIL":
                st.error(result.error)
                continue

            tab1, tab2, tab3, tab4 = st.tabs([
                "Extracted Text",
                "Extracted Data",
                "Guidelines",
                "Future Plan"
            ])

            with tab1:
                st.text_area(
                    "",
                    result.extracted_text,
                    height=300,
                    key=f"text_{result.filename}"
                )

            with tab2:
                st.write(result.extracted_data)

            with tab3:
                st.write(result.guideline_context)

            with tab4:
                st.write(result.final_plan)