# import streamlit as st
import pdfplumber
from app.backend.graph import clinical_agent  # Importing your working graph

def extract_tables_and_text(pdf_path):
    complete_text=""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            complete_text+="\n\nnewwpagee\n\n"
            
            # Extract plain text with layout preserved
            text = page.extract_text(layout=True)
            complete_text+=f"\n{text}"
            
            # Extract structured tables (e.g., Vitamin D, WBC counts)
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    complete_text+=f"\n{row}"  # Output as a list of strings representing cells
    return complete_text

# @st.cache_data(show_spinner=False)
def run_clinical_analysis(text):
    print("Running clinical analysis agent...")
    inputs = {
        "report_text": text,
        "extracted_data": "",
        "guideline_context": "",
        "final_plan": ""
    }
    print("inputs prepared for agent:")
    return clinical_agent.invoke(inputs)