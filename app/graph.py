import os
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
# 1. Imports from langchain-community (Requires: pip install langchain-community rank_bm25)
from langchain_community.retrievers import BM25Retriever

# 2. Imports from core langchain 
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# The correct path for the document compressor
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_compressors import FlashrankRerank


print("Initializing Ollama Embeddings and ChromaDB client...")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# print("Initializing Ollama Embeddings and ChromaDB client...")
print(f"CHROMA_HOST: {CHROMA_HOST}, CHROMA_PORT: {CHROMA_PORT}, OLLAMA_URL: {OLLAMA_URL}")
# 1. Define the State
class AgentState(TypedDict):
    report_text: str           # Raw text from the uploaded PDF
    extracted_data: str        # Structured JSON-like lab results
    optimized_extracted_data: str  # Optimized search query
    guideline_context: str     # Results found in your ChromaDB
    final_plan: str            # The final referral/follow-up draft

# 2. Initialize the Model (Pointing to your Ollama Docker address)
# Ensure you have 'llama3' pulled in your Ollama container
llm = ChatOllama(
    model="llama3", 
    base_url="http://localhost:11434"
)
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_client = Chroma(client=client, collection_name="local_rag", embedding_function=embeddings)
# collection = client.get_collection("local_rag")
# data = collection.get(include=["documents"])
# print("ChromaDB collection 'local_rag' documents:", data["documents"])

#chroma retriever
chroma_retriever = chroma_client.as_retriever(search_kwargs={"k": 8})
print("Chroma retriever initialized with k=8: ", chroma_retriever)
#bm25 retriever
bm25_documents = [Document(page_content=text) for text in chroma_client._collection.get(include=["documents"])["documents"]]
bm25_retriever = BM25Retriever.from_documents(bm25_documents)
bm25_retriever.k = 6
print("BM25 retriever initialized with k=6 and Chroma retriever with k=8.")
# Combine the two retrievers into an ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], 
    weights=[0.6, 0.4])

#reranker
reranker = FlashrankRerank(top_n=5)


# --- NODES ---

def extract_labs_node(state: AgentState):
    """Extracts only numerical lab values and key findings from the messy report."""
    print("Extracting lab values from report...")
    print("Report text preview:", state['report_text'] + "...")
    prompt = f"""
    You are a medical data extractor. Extract only the lab values (e.g. HbA1c, LDL) 
    from the following text. Ignore all other noise. give the output in a structured format (Lab Name: Value (Unit)).and nothing else. if input is not valid then return empty string.
    REPORT: {state['report_text']}
    OUTPUT FORMAT: Lab Name: Value (Unit)
    """
    response = llm.invoke(prompt)
    print("Extracted lab values:", response.content)
    return {"extracted_data": response.content}

def optimize_extract_labs_node(state: AgentState):
    """optimize the extracted lab values for better search results."""
    print("Optimizing extracted lab values...")
    prompt = f"""
    You are an expert medical search optimizer. Based on these extracted patient report details,
    write a single concise search query (2-3 sentences) designed to find the exact relevant WHO 
    diagnostic guidelines or nutrient requirements in a textbook database. Do not include patient names or raw numbers.
    Extracted Details: {state['extracted_data']}
    Optimized Search Query: "only optimized query here"
    """
    response = llm.invoke(prompt)
    print("optimized Extracted lab values:", response.content)
    return {"optimized_extracted_data": response.content}

def search_chroma_node(state: AgentState):
    """
    This is where you bridge to your existing RAG.
    For now, we simulate the tool call. You will replace the 'retriever' 
    call with your specific ChromaDB tool.
    """
    print("Searching ChromaDB for relevant guidelines...")
    # 1. Take the extracted lab value (e.g., HbA1c: 8.5)
    query = f"WHO guidelines and clinical thresholds for {state['extracted_data']}"
    # 2. Perform actual similarity search in your 'local_rag' collection
    # k=2 means get the top 2 most relevant paragraphs from the WHO PDF
    # docs = chroma_client.similarity_search(query, k=8)
    print("Querying ChromaDB with:", query)
    
    #-----------------------------------------------------------------#
    # compressor = LLMChainExtractor.from_llm(llm)
    # print("Ensemble retriever and LLMChainExtractor initialized.")
    # # Wrap your ensemble retriever with the compressor
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, 
    #     base_retriever=ensemble_retriever)
    # print("ContextualCompressionRetriever initialized.")
    # # Execute the final pipeline using your optimized query
    final_documents = ensemble_retriever.invoke(query)

    # Step 1: retrieve candidates
    # docs = ensemble_retriever.invoke(query)

    # Step 2: rerank (FAST)
    reranked_docs = reranker.compress_documents(final_documents, query)

    print(f"Reranked to {len(reranked_docs)} docs")

    # Step 3: take top results
    docs= reranked_docs[:5]

    # print(f"Retrieved {len(final_documents)} documents from ChromaDB after compression.")
    #------------------------------------------------------------------#
    # Limit to top 2 paragraphs for your final prompt
    # docs = final_documents[:5]
    # 3. Join the document content into one string for the next LLM node
    retrieved_content = "\n".join([doc.page_content for doc in docs])
    print("Retrieved guidelines:", retrieved_content[:100] + "...")

    # TODO: Connect this to your existing ChromaDB retrieval logic
    # example_context = "WHO Threshold for HbA1c: >7.0 is Type 2 Diabetes. Action: Specialist Referral."
    
    return {"guideline_context": retrieved_content}

def draft_plan_node(state: AgentState):
    """Compares Labs vs Guidelines and drafts the final action."""
    print("Drafting final plan...")
    print(f"Comparing extracted data: {state['extracted_data']} \n with guidelines: {state['guideline_context']}")
    prompt = f"""
    As a clinical assistant, compare the patient's labs with the guidelines.
    PATIENT LABS: {state['extracted_data']}
    WHO GUIDELINES: {state['guideline_context']}
    
    TASK: Draft a concise referral or follow-up plan. If labs are normal, state no action.
    """
    response = llm.invoke(prompt)
    print("Drafted final plan:", response.content)
    return {"final_plan": response.content}

# --- GRAPH CONSTRUCTION ---
# print("extracted node:", extract_labs_node(agent_state := AgentState(report_text="Patient has elevated HbA1c levels.", extracted_data="", guideline_context="", final_plan="")))
workflow = StateGraph(AgentState)
print("Building the clinical analysis workflow...")

# Add Nodes
workflow.add_node("extractor", extract_labs_node)
print("Added extractor node.")
workflow.add_node("optimizer", optimize_extract_labs_node)
print("Added optimizer node.")
workflow.add_node("researcher", search_chroma_node)
print("Added researcher node.")
workflow.add_node("writer", draft_plan_node)
print("Added writer node.")

# Define Edges (The flow)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "optimizer")
workflow.add_edge("optimizer", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

# Compile the Graph
clinical_agent = workflow.compile()
# clinical_agent.invoke({"report_text": "Patient has elevated HbA1c levels.", "extracted_data": "", "guideline_context": "", "final_plan": ""})