import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import chromadb

# Configuration from Environment Variables (for Docker)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://docker.internal")

# Initialize global components
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
db = Chroma(client=client, collection_name="local_rag", embedding_function=embeddings)
llm = Ollama(base_url=OLLAMA_URL, model="llama3")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    try:
        client.heartbeat()
        count = db._collection.count()
        print(f"✅ Connected to ChromaDB. Current index size: {count} chunks.")
        if count == 0:
            print("⚠️ WARNING: Database is empty. Run ingest.py to load data.")
    except Exception as e:
        print(f"❌ Critical Error: Could not connect to ChromaDB: {e}")
    
    yield # The app runs here
    
    # --- SHUTDOWN LOGIC ---
    print("Shutting down Researcher Agent...")

app = FastAPI(lifespan=lifespan)

class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_rag(question: Question):
    docs = db.similarity_search(question.text, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {question.text}\nAnswer:"
    response = llm.invoke(prompt)
    return {"answer": response, "sources": [doc.metadata for doc in docs]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
