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

from testing.base_unit_conversion import update_unit, convert


print("Initializing Ollama Embeddings and ChromaDB client...")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# print("Initializing Ollama Embeddings and ChromaDB client...")
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
        print(f"Reading {len(pdf.pages)} pages...")
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

    return "\n\n<NEW_PAGE>\n\n".join(pages)

# --- NODES ---

def extract_labs_node(state: AgentState):

    """
    Gemini extracts all laboratory values from the report.

    Output:
        extracted_data -> Python list
    """

    print("=" * 60)
    print("STEP 1 : Extracting laboratory values...")
    print("=" * 60)

    report = state["report_text"]
    # print(report)

    prompt = f"""
Analyze the following medical report. Identify abnormal values and key clinical findings. Do not invent information. If no abnormalities are found, state that.
Compare values and reference range numerically as float values.
Return ONLY valid JSON in this exact format:

{{
  "summary": "<brief summary>",
  "abnormal_findings": [
    {{
      "finding": "<finding>",
      "lab_name": "<lab name> (eg. Absolute Lymphocyte Count, RBC Count, Total WBC Count)"
      "value" : <lab value> if given else None,
      "unit" : <unit>,
      "explanation": "<brief explanation>"
    }}
  ]
  }}

Medical Report:
{report}
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You analyze report text and generate appropriate follow-up plan and reffer specialist "
            ),
            HumanMessage(content=prompt)
        ]
    )
    
    data = json.loads(response.content)
    print(data)
    # data["value"],data["unit"]=convert(data["value"],data["unit"])
    if not data["abnormal_findings"]:
        return 
    for i in data["abnormal_findings"]:
        print(f"converting {i["value"]},{i["unit"]} to {convert(i["value"],i["unit"])}")
        i["value"],i["unit"]=convert(i["value"],i["unit"])
    return {"extracted_data": data}

def retrieve_guidelines_node(state: AgentState):
    """
    Retrieve guideline documents for each abnormal laboratory finding.
    """

    print("=" * 60)
    print("STEP 2 : Retrieving Guidelines...")
    print("=" * 60)

    extracted = state["extracted_data"]

    findings = extracted.get("abnormal_findings", [])

    if not findings:
        print("No abnormal findings.")
        return {
            "guideline_context": [
                {"content": "No abnormal findings, so no guidelines were retrieved."}
            ]
        }

    # ---------------------------------------
    # Deduplicate lab names
    # ---------------------------------------

    unique_labs = {
        f["lab_name"].strip().lower(): {"lab_name":f["lab_name"],
                                        "value":f["value"]}
        for f in findings
        if f.get("lab_name")
    }

    print(f"Unique labs: {list(unique_labs.values())}")

    all_documents = []

    # ---------------------------------------
    # Retrieve guidelines
    # ---------------------------------------

    chroma_retriever = chroma_client.as_retriever(
        search_kwargs={"k": 2}
    )

    for lab in unique_labs.values():
        print(f"Searching: {lab["lab_name"]}")

        docs = chroma_retriever.invoke(lab["lab_name"])
        if docs:
            for doc in docs:
                print(f"from {doc.page_content}")
                doc.page_content=update_unit(doc.page_content,lab)
                print(f"to {doc.page_content}")
                
        all_documents.extend(docs)

    if not all_documents:
        return {
            "guideline_context": [
                {"content": "No guidelines retrieved."}
            ]
        }

    # ---------------------------------------
    # Remove duplicates
    # ---------------------------------------

    unique_docs = {}

    for doc in all_documents:
        unique_docs[doc.page_content] = doc

    documents = list(unique_docs.values())

    guideline_context = []

    for doc in documents:
        # print(f"updating {doc.page_content} to {update_unit(doc.page_content)}")
        
        guideline_context.append(
            {
                "content": doc.page_content
            }
        )

    print(f"Retrieved {len(guideline_context)} guideline documents")

    return {
        "guideline_context": guideline_context
    }

def clinical_reasoning_node(state: AgentState):
    """
    Final clinical reasoning.

    Input:
        extracted_data
        guideline_context

    Output:
        final_plan
    """

    print("=" * 60)
    print("STEP 3 : Clinical Reasoning...")
    print("=" * 60)

    extracted = state["extracted_data"]
    findings = extracted.get("abnormal_findings", [])

    if not findings:
        print("No abnormal findings.")
        return {
            "final_plan": {
                "summary": extracted.get("summary", "No significant abnormalities."),
                "specialist": "None",
                "follow_up_plan": "No abnormal findings were identified. Continue routine preventive healthcare and follow up with your primary care physician as recommended.",
                "disclaimer": "This is not a medical diagnosis. Please review these findings with your primary care physician."
            }
        }

    context = "\n\n".join(
        doc["content"] for doc in state["guideline_context"]
    )
    for doc in state["guideline_context"]:
        content=doc["content"]
        parts={}
        for item in content.split(", "):
            key, value = item.split(": ", 1)
            parts[key] = value
        if parts["status"].strip().lower()=="normal":
            return {"final_plan": [{"summary": "No significant abnormalities.",
                                    "specialist": "None",
                                    "follow_up_plan": "No abnormal findings were identified. Continue routine preventive healthcare and follow up with your primary care physician as recommended.",
                                    "disclaimer": "This is not a medical diagnosis. Please review these findings with your primary care physician."}]}


    prompt = f"""
You are an experienced clinical decision support assistant.

Patient findings:

{json.dumps(extracted, indent=2)}

Relevant clinical guidelines (prioritize these reference range if correct lab):

{context}

Return ONLY valid JSON in this exact format:

{{
  "summary": "<updated clinical summary>",
  "specialist": "<most appropriate specialist>",
  "follow_up_plan": "<50-100 word follow-up plan based on the guidelines>",
  "disclaimer": "This is not a medical diagnosis. Please review these findings and recommendations with your primary care physician."
}}

Rules:
- Base recommendations only on the provided findings and guideline context.
- Do not invent laboratory values or reference ranges.
- Recommend a single most appropriate specialist.
- If the guidelines are insufficient, make conservative general follow-up recommendations.
- Return ONLY JSON.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a medical clinical reasoning assistant."
            ),
            HumanMessage(content=prompt),
        ]
    )

    text = response.content

    if isinstance(text, list):
        text = "".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in text
        )

    text = text.strip()

    if text.startswith("```"):
        text = (
            text.replace("```json", "")
                .replace("```", "")
                .strip()
        )

    try:
        final_plan = json.loads(text)
    except json.JSONDecodeError as e:
        print(text)
        raise Exception(f"Model returned invalid JSON\n{e}")

    print("Clinical reasoning completed.")
    print(final_plan)

    return {"final_plan": final_plan}

workflow = StateGraph(AgentState)
print("Building the clinical analysis workflow...")

# Add Nodes
workflow.add_node("extractor", extract_labs_node)
print("Added extractor node.")
# workflow.add_node("optimizer", optimize_extract_labs_node)
# print("Added optimizer node.")
workflow.add_node("researcher", retrieve_guidelines_node)
print("Added researcher node.")
workflow.add_node("writer", clinical_reasoning_node)
print("Added writer node.")

# Define Edges (The flow)

workflow.set_entry_point("extractor")
# workflow.add_edge("extractor", "optimizer")
# workflow.add_edge("optimizer", "researcher")
workflow.add_edge("extractor", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

# Compile the Graph
clinical_agent = workflow.compile()

