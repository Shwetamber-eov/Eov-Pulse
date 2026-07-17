import os
import pandas as pd
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# -----------------------------
# Configuration
# -----------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

CSV_FILE = "thresholds_rag1.csv"
COLLECTION_NAME = "local_rag3"   #2 == table data

# -----------------------------
# Embeddings
# -----------------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_URL,
)

client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
)

vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# -----------------------------
# Read CSV
# -----------------------------
df = pd.read_csv(CSV_FILE).fillna("")
# df = pd.read_excel(CSV_FILE).fillna("")

print(f"Loaded {len(df)} rows")

documents = []

for _, row in df.iterrows():

    # searchable text
    text = ", ".join(
        f"{col}: {row[col]}"
        for col in df.columns
        if str(row[col]).strip() != ""
    )

    metadata = {
    "normalized_name": row["normalized_name"].strip().lower(),
    "biomarker_name": row["biomarker_name"].strip().lower(),
    "panel_name": row["panel_name"].strip().lower(),
    "demographic_group": row["demographic_group"].strip().lower(),
    }
    documents.append(
        Document(
            page_content=text,
            metadata=metadata
        )
    )

print(f"Prepared {len(documents)} documents")

# -----------------------------
# Clear collection (optional)
# -----------------------------
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# -----------------------------
# Insert
# -----------------------------
vectorstore.add_documents(documents)
print(f"Ingested {len(documents)} rows into '{COLLECTION_NAME}'")