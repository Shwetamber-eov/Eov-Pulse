import os
import chromadb
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Note: Use localhost here if running from your terminal, 
# but CHROMA_HOST if running inside the same Docker network.
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://docker.internal")

def run_ingestion():
    loader = DirectoryLoader("./data", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
    
    # Connect to the running Docker service
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    print(f"Uploading {len(chunks)} chunks to ChromaDB at {CHROMA_HOST}...")
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="local_rag"
    )
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()
