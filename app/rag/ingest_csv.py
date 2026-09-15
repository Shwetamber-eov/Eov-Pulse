import os
import pandas as pd
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from ingest_logger.rag_logger import log_ingestion
from pathlib import Path

# Gets the folder where your current script resides
ROOT = Path(__file__).resolve().parent.parent.parent

# -----------------------------
# Configuration
# -----------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

CSV_FILE = ROOT / "data" / "thresholds_rag2.csv"
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


try:

    df = pd.read_csv(CSV_FILE).fillna("")
    print(f"Loaded {len(df)} rows")
    documents = []
    for _, row in df.iterrows():
        text = ", ".join(
            f"{col}: {row[col]}"
            for col in df.columns
            if str(row[col]).strip() != ""
        )

        metadata = {
            # "normalized_name": row["normalized_name"].strip().lower(),
            "biomarker_name": row["biomarker_name"].strip().lower(),
            "panel_name": row["panel_name"].strip().lower(),
            "demographic_group": row["demographic_group"].strip().lower(),
            "lower_limit": row["lower_limit"],
            "upper_limit": row["upper_limit"],
            "unit": row["unit"].strip().lower()

        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    # try:
    #     client.delete_collection(COLLECTION_NAME)
    # except:
    #     pass

    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    vectorstore.add_documents(documents)

    # # Extract the full file name (with extension)
    # print(file_path.name)
    # # Output: ingest_log.jsonl

    # # Extract ONLY the file name (WITHOUT the extension)
    # print(file_path.stem)
    log_ingestion(
        document_name=CSV_FILE.name,
        status="SUCCESS",
        rows_processed=len(documents),
        collection_name=COLLECTION_NAME,
    )

except Exception as e:

    log_ingestion(
        document_name=CSV_FILE,
        status="FAILED",
        rows_processed=0,
        collection_name=COLLECTION_NAME,
        error=str(e),
    )

    raise