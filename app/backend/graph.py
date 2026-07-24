import os
import json
import re
import pdfplumber
import chromadb
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_ollama import OllamaEmbeddings
from rapidfuzz import fuzz
from testing.base_unit_conversion import convert,update_unit_with_status


print("Initializing Ollama Embeddings and ChromaDB client...")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# #print("Initializing Ollama Embeddings and ChromaDB client...")
print(f"CHROMA_HOST: {CHROMA_HOST}, CHROMA_PORT: {CHROMA_PORT}, OLLAMA_URL: {OLLAMA_URL}")
# 1. Define the State
class AgentState(TypedDict):
    report_text: str
    extracted_data: list
    guideline_context: list
    final_plan: list

llm = ChatOllama(
    model="gemma3:12b",
    base_url=OLLAMA_URL,
    temperature=0,
    num_predict=2048,
    num_ctx=4096,
    format="json"
)

embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_client = Chroma(client=client, collection_name="local_rag3", embedding_function=embeddings)


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

            # Extract Text
            text = page.extract_text(layout=True)
            if not text:
                text = page.extract_text()
            if text:
                page_content.append(text.strip())

            # Extract Tables
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

            # Save Page
            page_text = "\n".join(page_content).strip()
            if page_text:
                pages.append(page_text)

    return "\n\n<NEW_PAGE>\n\n".join(pages)


def extract_labs_node(state: AgentState):

    """
    Gemini extracts all laboratory values from the report.

    Output:
        extracted_data -> Python list
    """

    report = state["report_text"]

    prompt = f"""
You are an expert clinical laboratory extraction system.

Your job is ONLY to extract laboratory measurements.

Return ONLY valid JSON.

Schema:

[
    {{
        "test_name": "",
        "value": null if not given,
        "lower_limit": <number or null>  (If >10, set to 10. If between 100 and 1000, set to 100 )
        "upper_limit": <number or null>  (If <500, set to 500. If between 10 and 200, set to 200.)
        "unit": ""
        "status": "" (good or bad ) only if given else ""
    }}
]

Rules

1. Extract every laboratory test.
2. do not assume lab test if not mentioned
3. Ignore:
   - Diagnoses
   - Clinical notes
   - Medications
   - Doctor comments
   - Recommendations
   - Symptoms
   - Patient history

4. Keep original units.

5. value must always be numeric.

6. Return JSON only.

Patient Report:

{report}
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You extract laboratory values."
            ),
            HumanMessage(content=prompt)
        ]
    )

    labs=next(iter(json.loads(response.content).values()))
    for lab in labs:
        lab["value"],lab["unit"]=convert(lab["value"],lab["unit"])
    return {"extracted_data": labs}

def retrieve_guidelines_node(state: AgentState):
    """
    Retrieve guideline documents for each extracted laboratory test.
    """
    labs = state["extracted_data"]

    if not labs:
        print("No data extracted")
        return {"guideline_context": [{"content": "No data extracted so no guidelines can be retrieved"}]}

    # ---------------------------------------
    # Deduplicate lab names
    # ---------------------------------------

    unique_tests = {
        lab["test_name"].strip().lower(): {
            "lab_name": lab["test_name"],
            "value": lab["value"],
            "unit": lab["unit"],
            "lower_limit": lab["lower_limit"],
            "upper_limit": lab["upper_limit"],
            "status": lab["status"]
        }
        for lab in labs
        if lab.get("test_name")
    }

    # ---------------------------------------
    # Create retriever (no filtering)
    # ---------------------------------------

    chroma_retriever = chroma_client.as_retriever(
        search_kwargs={"k": 2}
    )

    all_documents = []

    # ---------------------------------------
    # Retrieve documents
    # ---------------------------------------

    for test in unique_tests.values():
        query = test["lab_name"]
        print(f"Searching: {query}")
        docs = chroma_retriever.invoke(query)
        all_documents.append(docs)

    if not all_documents:
        return {
            "guideline_context": [
                {"content": "No guidelines retrieved."}
            ]
        }

    #print(f"Retrieved {len(all_documents)} documents")
    
    # ---------------------------------------
    # Remove duplicates
    # ---------------------------------------

    unique_docs = {}
    documents=[]
    for lst in all_documents:
        for doc in lst:
            unique_docs[doc.page_content[:30]] = doc
        documents.append(list(unique_docs.values()))
        unique_docs={}


    #print(f"After deduplication: {len(documents)}")

    # ---------------------------------------
    # Build context
    # ---------------------------------------
    
    guideline_context = []
    for (lab, lst) in zip(unique_tests.values(), documents):
        for doc in lst:
            content,status,base_lower,base_upper=update_unit_with_status(doc.page_content,lab)
            if status=="normal":
                break
            lab["status"]=status
            guideline_context.append(
                        {
                            "content": content,
                            "lower":base_lower,
                            "upper":base_upper,
                            "for_lab": lab
                        }
                    )

    print(f"Final Context: {len(guideline_context)} documents")

    return {"guideline_context": guideline_context}

def clinical_reasoning_node(state: AgentState):
    """
    Final Gemini reasoning.

    Input:
        extracted_data
        guideline_context

    Output:
        final_plan
    """
    labs = state["extracted_data"]
    if len(labs)==0:
        return {"final_plan": [{"content":"no data extracted so no guidelines can be retrieved"}]}
    context1=state["guideline_context"]
    context2=""
    prompt=""
    context2 = "\n\n".join(f"Patient labs: {doc['for_lab']}\nwith guidelines: {doc['content']}" for doc in context1)

    if len(context2)==0:
        return {"final_plan": {"final_plan":"everything is normal", "specialist": None}}
    prompt = f"""
You are a clinical decision support assistant.

The laboratory status has already been determined.
Do NOT change or recalculate it.

Task

If status is "High" or "Low":
- Use the provided guideline.
- Recommend the single most appropriate specialist.
- Write a concise follow-up plan (50–100 words).
- Do not diagnose diseases.
- Do not invent information beyond the guideline.

If status is "Unknown":
- No matching guideline was found.
- Use the laboratory test, value, unit, panel, demographic group, and available context.
- Recommend the most appropriate specialist.
- Write a cautious follow-up plan.
- Mention uncertainty when appropriate.
- Do not diagnose diseases.

Return ONLY valid JSON.

Schema

{{
  "test_name": "",
  "patient_value": 0,
  "unit": "",
  "reference_range": "",
  "status": "",
  "specialist": "",
  "plan": ""
}}

Input

{context2}
"""
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a medical clinical reasoning assistant."
            ),
            HumanMessage(content=prompt)
        ]
    )
    final_plan=json.loads(response.content)

    return {"final_plan": final_plan}

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)
print("Building the clinical analysis workflow...")

# Add Nodes
workflow.add_node("extractor", extract_labs_node)
workflow.add_node("researcher", retrieve_guidelines_node)
workflow.add_node("writer", clinical_reasoning_node)

# Define Edges (The flow)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

# Compile the Graph
clinical_agent = workflow.compile()