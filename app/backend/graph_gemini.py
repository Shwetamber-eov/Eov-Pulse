import os
import json
import re
import pdfplumber
import chromadb
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (EnsembleRetriever,MergerRetriever)
from langchain_community.document_compressors import FlashrankRerank
from langchain_ollama import OllamaEmbeddings

load_dotenv()

#print("Initializing Ollama Embeddings and ChromaDB client...")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print(f"CHROMA_HOST: {CHROMA_HOST}, CHROMA_PORT: {CHROMA_PORT}, OLLAMA_URL: {OLLAMA_URL}")
# 1. Define the State
class AgentState(TypedDict):
    report_text: str
    extracted_data: list
    guideline_context: list
    final_plan: list

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_retries=3,
)

embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
# # chroma_client = Chroma(client=client, collection_name="local_rag", embedding_function=embeddings)
# chroma_client1 = Chroma(client=client, collection_name="local_rag2", embedding_function=embeddings)
# chroma_client2 = Chroma(client=client, collection_name="local_rag3", embedding_function=embeddings)
# # collection = client.get_collection("local_rag")
# # data = collection.get(include=["documents"])
# # #print("ChromaDB collection 'local_rag' documents:", data["documents"])

# #chroma retriever

# #if 1 collection
# # chroma_retriever= chroma_client.as_retriever(search_kwargs={"k": 8})
# #if 2 collections
# chroma_retriever1 = chroma_client1.as_retriever(search_kwargs={"k": 8})
# chroma_retriever2 = chroma_client2.as_retriever(search_kwargs={"k": 8})
# multi_collection_retriever = MergerRetriever(retrievers=[chroma_retriever1, chroma_retriever2])

# #print("Chroma retriever initialized with k=8: ")
# #bm25 retriever for 2 collection
# # bm25_documents = [Document(page_content=str(text)) if "Blood Test Normal Range" not in str(text) else Document(page_content=str(text[100:])) for text in (chroma_client._collection.get(include=["documents"])["documents"] for chroma_client in [chroma_client1, chroma_client2])]
# bm25_documents = [Document(page_content=doc) if "Blood Test Normal Range" not in doc else Document(page_content=doc[100:])  for chroma in [chroma_client1, chroma_client2] for doc in chroma._collection.get(include=["documents"])["documents"]]
# #print("bm25 initialized with documents from both collections.", len(bm25_documents))
# #bm25 for 1 collecion
# # bm25_documents = [Document(page_content=text) if "Blood Test Normal Range" not in text else Document(page_content=text[100:]) for text in chroma_client._collection.get(include=["documents"])["documents"]]
# bm25_retriever = BM25Retriever.from_documents(bm25_documents)
# bm25_retriever.k = 8
# #print("BM25 retriever initialized with k=8 and Chroma retriever with k=8.")
# # Combine the two retrievers into an ensemble retriever
# ensemble_retriever = EnsembleRetriever(
#     # retrievers=[bm25_retriever, chroma_retriever],                    #for 1 collection
#     retrievers=[bm25_retriever, multi_collection_retriever],            #for 2 collections
#     weights=[0.6, 0.4])

# #reranker
# reranker = FlashrankRerank(top_n=5)

def extract_tables_and_text(pdf_path: str) -> str:
    """
    Extract text and tables from a PDF while preserving page boundaries.
    Returns:
        str: Complete report text.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        #print(f"Reading {len(pdf.pages)} pages...")
        for page_no, page in enumerate(pdf.pages, start=1):
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

    #print("=" * 60)
    #print("STEP 1 : Extracting laboratory values...")
    #print("=" * 60)

    report = state["report_text"]

    prompt = f"""
You are a clinical decision support assistant.

Analyze the entire clinical report below.

Tasks:
1. Extract all laboratory test results.
2. Summarize the key laboratory and clinically relevant findings from the entire report.
3. Recommend the single most appropriate medical specialist, if indicated.
4. Generate a concise follow-up plan (50–100 words).

Rules:
- Analyze the complete report, including laboratory results, clinical notes, impressions, and recommendations.
- Extract only laboratory tests explicitly mentioned.
- Keep laboratory values and units exactly as written.
- Use numeric values only. If a value is missing, use null.
- Do not invent laboratory values, reference ranges, or diagnoses.
- Base the summary, specialist recommendation, and follow-up plan only on information present in the report.
- If no specialist referral is indicated, set specialist to "None".
- If the report does not contain enough information for a recommendation, state this in the follow-up plan.

Return ONLY valid JSON.

Schema:
[
{{
  "summary": "",
  "specialist": "",
  "follow_up_plan": ""
}}
]

Clinical Report:

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

    if isinstance(response.content, list):
        text = "".join(
        block["text"] if isinstance(block, dict) else str(block)
        for block in response.content
    )
    else:
        text = response.content

    text = text.strip()


    # Gemini sometimes wraps JSON
    if text.startswith("```"):

        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:

        labs = json.loads(text)

    except Exception as e:

        #print(text)

        raise Exception(
            f"Invalid JSON returned by Gemini\n{e}"
        )

    #print(f"Extracted {len(labs)} laboratory tests.", labs)

    return {

        "final_plan": labs

    }

def retrieve_guidelines_node(state: AgentState):
    """
    Retrieve guideline documents for each extracted laboratory test.

    Uses:
        BM25
        Chroma
        FlashRank

    Returns:
        guideline_context -> list[str]
    """

    #print("=" * 60)
    #print("STEP 2 : Retrieving Guidelines...")
    #print("=" * 60)

    labs = state["extracted_data"]
    all_documents = []

    # ---------------------------------------
    # Deduplicate labs before querying
    # ---------------------------------------
    #print(type(labs[0]["test_name"]))
    #print((labs[0]["test_name"]))
    
    unique_tests = {
        lab["test_name"].strip().lower(): lab["test_name"]
        for lab in labs
    }

    # ---------------------------------------
    # Retrieve documents for each lab
    # ---------------------------------------

    for _, test_name in unique_tests.items():
        query = f"{test_name}"
        #print(f"Searching: {query}")
        docs = ensemble_retriever.invoke(query)
        all_documents.extend(docs)

    #print(f"\nRetrieved {len(all_documents)} documents")

    # ---------------------------------------
    # Remove duplicate documents
    # ---------------------------------------

    unique_docs = {}

    for doc in all_documents:

        unique_docs[doc.page_content] = doc

    documents = list(unique_docs.values())

    #print(f"After deduplication : {len(documents)}")

    # ---------------------------------------
    # FlashRank
    # ---------------------------------------

    reranked = reranker.compress_documents(

        documents,

        json.dumps(labs)

    )

    #print(f"After reranking : {len(reranked)}")

    # ---------------------------------------
    # Keep Top Results
    # ---------------------------------------

    top_docs = reranked[:5]

    guideline_context = []

    for doc in top_docs:

        guideline_context.append(

            {
                "content": doc.page_content,

                # "metadata": doc.metadata
            }

        )

    #print(f"Final Context : {len(guideline_context)} documents {guideline_context}")

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

    #print("=" * 60)
    #print("STEP 3 : Clinical Reasoning...")
    #print("=" * 60)

    labs = state["extracted_data"]

    context = "\n\n".join(
        doc["content"]
        for doc in state["guideline_context"]
    )

    prompt = f"""
You are an experienced clinical decision support assistant.

Patient laboratory values:

{json.dumps(labs, indent=2)}

Clinical guideline context:

{context}

--------------------------------------------------

Return ONLY valid JSON.

Schema

[
  {{
    "test_name":"",
    "patient_value":0,
    "unit":"",
    "reference_range":"",
    "status":"",
    "specialist":"",
    "plan":""
  }}
]

--------------------------------------------------

Rules

1. Match each laboratory test with the guideline.

2. Extract the correct reference range.

3. Compare numerically.

4. Status must be one of:

Normal

Low

High

Unknown

5. If Normal

specialist = "None"

plan = "No action required."

6. If abnormal

recommend the most appropriate specialist.

write a concise follow-up plan.

7. If guideline is unavailable

status = Unknown

8. Never invent reference ranges.

9. Return ONLY JSON.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a medical clinical reasoning assistant."
            ),
            HumanMessage(content=prompt)
        ]
    )
    if isinstance(response.content, list):
        text = "".join(
        block["text"] if isinstance(block, dict) else str(block)
        for block in response.content
    )
    else:
        text = response.content

    text = text.strip()
    if text.startswith("```"):
        text = (text.replace("```json", "").replace("```", "").strip())

    try:

        final_plan = json.loads(text)

    except Exception as e:

        #print(text)

        raise Exception(
            f"Gemini returned invalid JSON\n{e}"
        )

    #print(f"Generated {len(final_plan)} assessments.")
    return {"final_plan": final_plan}

# --- GRAPH CONSTRUCTION ---
# #print("extracted node:", extract_labs_node(agent_state := AgentState(report_text="Patient has elevated HbA1c levels.", extracted_data="", guideline_context="", final_plan="")))
workflow = StateGraph(AgentState)
#print("Building the clinical analysis workflow...")

# Add Nodes
workflow.add_node("extractor", extract_labs_node)
#print("Added extractor node.")
# workflow.add_node("optimizer", optimize_extract_labs_node)
# #print("Added optimizer node.")
# workflow.add_node("researcher", retrieve_guidelines_node)
# #print("Added researcher node.")
# workflow.add_node("writer", clinical_reasoning_node)
# #print("Added writer node.")

# Define Edges (The flow)

workflow.set_entry_point("extractor")
# workflow.add_edge("extractor", "optimizer")
# workflow.add_edge("optimizer", "researcher")
# workflow.add_edge("extractor", "researcher")
# workflow.add_edge("researcher", "writer")
# workflow.add_edge("writer", END)
workflow.add_edge("extractor", END)

# Compile the Graph
clinical_agent = workflow.compile()

# clinical_agent.invoke({"report_text": extract_tables_and_text("Abhishek_health_report_1.pdf"), "extracted_data": "", "guideline_context": "", "final_plan": ""})