import streamlit as st
import pdfplumber
# from app.backend.graph_gemini import clinical_agent  # Importing your working graph
# from app.backend.graph_groq import clinical_agent  # Importing your working graph
# from app.backend.graph_gemma import clinical_agent  # Importing your working graph
# from app.backend.graph_phi import clinical_agent  # Importing your working graph
# from app.backend.graph_llama3_1 import clinical_agent  # Importing your working graph
# from app.backend.graph_simple import clinical_agent  # Importing your working graph
# from app.backend.graph_copy import clinical_agent  # Importing your working graph
from app.backend.graph_multiagent import clinical_agent, extract_tables_and_text  # Importing your working graph

def extract_tables_and_text(pdf_path: str) -> str:
    """
    Extract text and tables from a PDF while preserving page boundaries.
    Returns:
        str: Complete report text.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for _, page in enumerate(pdf.pages, start=1):
            page_content = []
            # -------------------------
            # Extract Text
            # -------------------------
            text = page.extract_text(layout=True)
            if not text:
                text = page.extract_text()
            if text:
                page_content.append(text.strip())
            # -------------------------
            # Extract Tables
            # -------------------------
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if not row:
                        continue
                    cleaned_row = []
                    for cell in row:
                        if cell is None:
                            cleaned_row.append("")
                        else:
                            cleaned_row.append(str(cell).replace("\n", " ").strip())
                    page_content.append(" | ".join(cleaned_row))
            # -------------------------
            # Save Page
            # -------------------------
            page_text = "\n".join(page_content).strip()
            if page_text:
                pages.append(page_text)

    return pages

# Helper function to "Freeze" AI results
# This prevents rerunning the agent when you click the button
@st.cache_data(show_spinner=False)
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