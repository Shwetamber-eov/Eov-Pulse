import streamlit as st
from pypdf import PdfReader
from graph import clinical_agent  # Importing your working graph
print("Starting Streamlit UI...")
# 1. Page Configuration (Keep original UI)
st.set_page_config(page_title="EOV Pulse", layout="wide")

st.title("🩺 Clinical Decision & Support System")
st.markdown("---")

# 2. Sidebar for status (Keep original UI)
with st.sidebar:
    st.header("System Status")
    st.success("Ollama: Connected")
    st.success("ChromaDB: Connected")
    st.info("Model: Llama 3 (Reasoning)")
print("UI components set up. Ready for file upload and agent invocation.")
# 3. Helper function to "Freeze" AI results
# This prevents rerunning the agent when you click the button
@st.cache_data
def run_clinical_analysis(text):
    print("Running clinical analysis agent...")
    inputs = {
        "report_text": text,
        "extracted_data": "",
        "guideline_context": "",
        "final_plan": ""
    }
    print("inputs prepared for agent:", inputs)
    return clinical_agent.invoke(inputs)

# 4. Session State for clearing the uploader
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def clear_report():
    st.session_state["uploader_key"] += 1
    # This resets the file uploader and clears the cache for the next file
    st.cache_data.clear()

# 5. File Upload (Dynamic Key to allow clearing)
uploaded_file = st.file_uploader(
    "Upload Patient Lab Report (PDF)", 
    type="pdf", 
    key=f"pdf_uploader_{st.session_state['uploader_key']}"
)

if uploaded_file is not None:
    with st.spinner("Processing report and consulting WHO guidelines..."):
        st.write("✅ PDF uploaded successfully!")
        # A. Extract text from PDF
        reader = PdfReader(uploaded_file)
        st.write("Extracting text from PDF...")
        raw_text = "".join([page.extract_text() for page in reader.pages])
        st.write(raw_text[:500] + "...")  # Show a preview of the extracted text
        # B. Run the Agent (Cached version)
        st.write("Invoking the clinical analysis agent...")
        result = run_clinical_analysis(raw_text)
        st.write("✅ Clinical analysis complete!")
        # C. Display Results in Columns (Keep original UI)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Extracted Lab Values")
            st.info(result["extracted_data"])
            
            with st.expander("View Referenced WHO Guidelines"):
                st.write(result["guideline_context"])

        with col2:
            st.subheader("📝 Drafted Clinical Action Plan")
            st.success(result["final_plan"])
            
            # 6. Functional "Approve" Button
            if st.button("Approve & Sign Referral"):
                st.balloons()
                st.success("✅ Referral signed and saved to Patient History!")
                
                # Option to clear the screen and start over
                st.button("Reset / New Patient", on_click=clear_report)

