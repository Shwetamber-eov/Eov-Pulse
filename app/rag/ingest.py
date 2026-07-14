import os
import re
import chromadb
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from app.rag.keywords import MEDICAL_KEYWORDS, JUNK_PATTERNS

# Note: Use localhost here if running from your terminal, 
# but CHROMA_HOST if running inside the same Docker network.
print("Starting data ingestion...")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def run_ingestion():
    target_path = os.path.join("..", "data")
    print(f"Loading PDF files from {target_path}...")
    loader = DirectoryLoader(target_path, glob="*.pdf", loader_cls=PDFPlumberLoader)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
    
    # Connect to the running Docker service
    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    # try:                                            #to clear the existing collection before ingesting new data
    #     client.delete_collection("local_rag")
    #     print("Cleared existing collection.")
    # except Exception:
    #     print("Collection did not exist yet. Creating fresh.")


    important_documents = []
    print(chunks[0].page_content, "...")
    for chunk in chunks:
        text = chunk.page_content
        text_lower = text.lower()
    
        # Rule A: Skip pure junk/citations using fast regex
        if any(re.search(pattern, text_lower) for pattern in JUNK_PATTERNS):
            continue
        
        # Rule B: Calculate keyword density (Count how many unique target words appear)
        words_found = [word for word in MEDICAL_KEYWORDS if word in text_lower]
    
        # Rule C: Strict structural check (Skip chunks that are mostly numbers/symbols like big tables)
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text) if len(text) > 0 else 1
        alpha_ratio = alpha_chars / total_chars
    
        # Triage: Keep it only if it contains at least 1 unique core keyword 
        # AND is mostly actual prose text (not a raw data table)
        if len(words_found) >= 2 and alpha_ratio > 0.30:
            # Prepend Nomic search prefix for best vector matching performance
            chunk.page_content = f"search_document: {chunk.page_content}"
            important_documents.append(chunk)

    print(f"Instantly filtered {len(chunks)} down to {len(important_documents)} important chunks.")
    print("Sample important chunk content:", [important_documents[x].page_content for x in range(len(important_documents))], "...")
    print(f"Uploading {len(important_documents)} chunks to ChromaDB at {CHROMA_HOST}...")

    
    db = Chroma.from_documents(
        documents=important_documents,
        embedding=embeddings,
        client=client,
        collection_name="local_rag2"
    )
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()
