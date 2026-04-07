import os
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# 1. Define the State
class AgentState(TypedDict):
    report_text: str           # Raw text from the uploaded PDF
    extracted_data: str        # Structured JSON-like lab results
    guideline_context: str     # Results found in your ChromaDB
    final_plan: str            # The final referral/follow-up draft

# 2. Initialize the Model (Pointing to your Ollama Docker address)
# Ensure you have 'llama3' pulled in your Ollama container
llm = ChatOllama(
    model="llama3", 
    base_url="http://host.docker.internal:11434"
)
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_client = Chroma(client=client, collection_name="local_rag", embedding_function=embeddings)

# --- NODES ---

def extract_labs_node(state: AgentState):
    """Extracts only numerical lab values and key findings from the messy report."""
    prompt = f"""
    You are a medical data extractor. Extract only the lab values (e.g. HbA1c, LDL) 
    from the following text. Ignore all other noise.
    REPORT: {state['report_text']}
    OUTPUT FORMAT: Lab Name: Value (Unit)
    """
    response = llm.invoke(prompt)
    return {"extracted_data": response.content}

def search_chroma_node(state: AgentState):
    """
    This is where you bridge to your existing RAG.
    For now, we simulate the tool call. You will replace the 'retriever' 
    call with your specific ChromaDB tool.
    """
    # 1. Take the extracted lab value (e.g., HbA1c: 8.5)
    query = f"WHO guidelines and clinical thresholds for {state['extracted_data']}"
    # 2. Perform actual similarity search in your 'local_rag' collection
    # k=2 means get the top 2 most relevant paragraphs from the WHO PDF
    docs = chroma_client.similarity_search(query, k=2)

    # 3. Join the document content into one string for the next LLM node
    retrieved_content = "\n".join([doc.page_content for doc in docs])
    
    # TODO: Connect this to your existing ChromaDB retrieval logic
    # example_context = "WHO Threshold for HbA1c: >7.0 is Type 2 Diabetes. Action: Specialist Referral."
    
    return {"guideline_context": retrieved_content}

def draft_plan_node(state: AgentState):
    """Compares Labs vs Guidelines and drafts the final action."""
    prompt = f"""
    As a clinical assistant, compare the patient's labs with the guidelines.
    PATIENT LABS: {state['extracted_data']}
    WHO GUIDELINES: {state['guideline_context']}
    
    TASK: Draft a concise referral or follow-up plan. If labs are normal, state no action.
    """
    response = llm.invoke(prompt)
    return {"final_plan": response.content}

# --- GRAPH CONSTRUCTION ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("extractor", extract_labs_node)
workflow.add_node("researcher", search_chroma_node)
workflow.add_node("writer", draft_plan_node)

# Define Edges (The flow)
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

# Compile the Graph
clinical_agent = workflow.compile()
