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
    print("after prompt")

    response = llm.invoke(
        [
            SystemMessage(
                content="You extract laboratory values."
            ),
            HumanMessage(content=prompt)
        ]
    )

    # if isinstance(response.content["result"], list):
    #     text = "".join(
    #     block["text"] if isinstance(block, dict) else str(block)
    #     for block in response.content
    # )
    # else:
    #     text = response.content["result"]

    # text = text.strip()


    # # Gemini sometimes wraps JSON
    # if text.startswith("```"):

    #     text = (
    #         text
    #         .replace("```json", "")
    #         .replace("```", "")
    #         .strip()
    #     )

    # try:

    #     labs = json.loads(text)

    # except Exception as e:

    #     print(text)

    #     raise Exception(
    #         f"Invalid JSON returned by Gemini\n{e}" )
    print(response.content)
    print(type(response.content))
    # print(json.loads(response.content))
    labs=next(iter(json.loads(response.content).values()))
    # print(f"labs {labs} type: {type(labs)}")
    print(f"Extracted {len(labs)} laboratory tests type {type(labs)} and content.", labs)
    for lab in labs:
        lab["value"],lab["unit"]=convert(lab["value"],lab["unit"])
    # print(labs)
    return {"extracted_data": labs}

def retrieve_guidelines_node(state: AgentState):
    """
    Retrieve guideline documents for each extracted laboratory test.
    """

    print("=" * 60)
    print("STEP 2 : Retrieving Guidelines...")
    print("=" * 60)

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
    print("="*10)
    print(f"Unique tests: {list(unique_tests.values())}")
    print("="*10)

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

    print(f"Retrieved {len(all_documents)} documents")
    # print("all docs: ",all_documents)
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


    print(f"After deduplication: {len(documents)}")

    # ---------------------------------------
    # Build context
    # ---------------------------------------
    # print("--"*10)
    # print([i, (lab,lst)] for i,(lab,lst) in enumerate(zip(unique_tests.values(),documents)))
    # print("--"*10)
    guideline_context = []
    for i, (lab, lst) in enumerate(zip(unique_tests.values(), documents)):
        # print(f"lst is: {lst} and j: {i}")  # Since j == i, we just use i
        # print(f"lab is: {lab} and i: {i}")
        
        for doc in lst:

            print("metadata: ",doc.metadata)
            print("doc: ", doc)
            content,status,base_lower,base_upper=update_unit_with_status(doc.page_content,lab)
            if status=="normal":
                break
                    # if lab["status"].strip().lower()=="good":
                    #     lab["status"]="normal"
                    # elif lab["status"].strip().lower()=="bad":
                    #     lab["status"]="bad"
                    # else:
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

    return {
        "guideline_context": guideline_context
    }

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
    print(len(labs))
    if len(labs)==0:
        print("no data extracted")
        return {"final_plan": [{"content":"no data extracted so no guidelines can be retrieved"}]}
    print("=" * 60)
    print("STEP 3 : Clinical Reasoning...")
    print("=" * 60)

    print("labs stored")
    context1=state["guideline_context"]
    context2=""
    prompt=""
    print("before prompt")
    # print(context1)
        # print(doc)
        # if doc["for_lab"]["status"]=="normal":
        #     print("status is normal")
        #     continue
    context2 = "\n\n".join(f"Patient labs: {doc['for_lab']}\nwith guidelines: {doc['content']}" for doc in context1)
    # print("status is",doc["for_lab"]["status"])
        # if doc["for_lab"]["status"]=="unknown":
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
#     prompt = f"""
# You are an experienced clinical decision support assistant.

# Extracted lab test along with its retrieved guidelines:

# {context}

# --------------------------------------------------

# Return ONLY valid JSON.

# Schema

# [
#   {{
#     "test_name":"",
#     "patient_value":0,
#     "unit":"",
#     "reference_range":"" (unit),
#     "status":"",
#     "specialist":"",
#     "plan":""
#   }}
# ]


# Rules

# 1. Match each laboratory test with the guideline.

# 2. Extract the correct reference range and critical range.

# 3. Compare numerically.

# 4. Status must be one of:

# Normal

# Low

# High

# Unknown

# 5. If Normal

# specialist = "None"

# plan = "No action required."

# 6. If abnormal

# recommend the most appropriate specialist (eg. Hematologist, Cardiologists).

# write a concise follow-up plan (50-100 words).

# 7. If guideline is unavailable

# status = Unknown

# 8. Never invent reference ranges.

# 9. Return ONLY JSON.
# """
    print("before passing prompt")
    # print(f"passing extracted data {state['extracted_data']} \nwith guideline context {context2}")
    # print("prompt is: ",prompt)
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a medical clinical reasoning assistant."
            ),
            HumanMessage(content=prompt)
        ]
    )
    # if isinstance(response.content, list):
    #     text = "".join(
    #     block["text"] if isinstance(block, dict) else str(block)
    #     for block in response.content
    # )
    # else:
    #     text = response.content

    # text = text.strip()
    # if text.startswith("```"):
    #     text = (text.replace("```json", "").replace("```", "").strip())

    # try:

    #     final_plan = json.loads(text)

    # except Exception as e:

    #     print(text)

    #     raise Exception(
    #         f"Gemini returned invalid JSON\n{e}"
    #     )
    print("after prompt")
    # print(response.content)
    final_plan=json.loads(response.content)
    # final_plan=json.loads("{'a':'b','c':'d'}")

    # print(f"Generated {len(final_plan)} assessments.", final_plan)
    return {"final_plan": final_plan}

# --- GRAPH CONSTRUCTION ---
# print("extracted node:", extract_labs_node(agent_state := AgentState(report_text="Patient has elevated HbA1c levels.", extracted_data="", guideline_context="", final_plan="")))
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

# clinical_agent.invoke({"report_text": extract_tables_and_text("Abhishek_health_report_1.pdf"), "extracted_data": "", "guideline_context": "", "final_plan": ""})