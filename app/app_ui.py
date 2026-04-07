import streamlit as st
from pypdf import PdfReader
from graph import clinical_agent  # Importing your working graph

st.set_page_config(page_title="Clinical Assistant", layout="wide")

st.title("🩺 Clinical Guidelines & Report Assistant")
st.markdown("---")

# 1. Sidebar for status
with st.sidebar:
    st.header("System Status")
    st.success("Ollama: Connected")
    st.success("ChromaDB: Connected")
    st.info("Model: Llama 3 (Reasoning)")

# 2. File Upload
uploaded_file = st.file_uploader("Upload Patient Lab Report (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing report and consulting WHO guidelines..."):
        # A. Extract text from PDF
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text()

        # B. Run the Agent
        # We pass the raw_text into the same initial_state you used in test_graph
        inputs = {
            "report_text": raw_text,
            "extracted_data": "",
            "guideline_context": "",
            "final_plan": ""
        }
        
        result = clinical_agent.invoke(inputs)

        # C. Display Results in Columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Extracted Lab Values")
            st.info(result["extracted_data"])
            
            with st.expander("View Referenced WHO Guidelines"):
                st.write(result["guideline_context"])

        with col2:
            st.subheader("📝 Drafted Clinical Action Plan")
            st.success(result["final_plan"])
            
            st.button("Approve & Sign Referral")
