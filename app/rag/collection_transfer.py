import chromadb
import os

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

print(f"CHROMA_HOST: {CHROMA_HOST}, CHROMA_PORT: {CHROMA_PORT}, OLLAMA_URL: {OLLAMA_URL}")

# 1. Connect to your persistent database
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# 2. Get the source collection
source_collection = client.get_collection(name="local_rag3")

# 3. Fetch ALL data (including documents, metadatas, and embeddings)
# Passing ids=None fetches everything up to the database limit
source_data = source_collection.get(
    include=["documents", "metadatas", "embeddings"]
)

# 4. Create or get your target collection
target_collection = client.get_or_create_collection(name="local_rag4")

# 5. Add the data to the target collection (source_collection remains untouched)
if source_data["ids"]:
    target_collection.add(
        ids=source_data["ids"],
        embeddings=source_data["embeddings"],
        metadatas=source_data["metadatas"],
        documents=source_data["documents"]
    )
    print(f"Successfully copied {len(source_data['ids'])} items to the new collection!")
else:
    print("Source collection is empty.")
