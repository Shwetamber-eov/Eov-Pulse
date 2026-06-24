import os
import re
import pdfplumber
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION_NAME = "local_rag"

client=chromadb.HttpClient(host=CHROMA_HOST,port=CHROMA_PORT)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def ingest_table():
    target_path = os.path.join("..", "data")
    pdf_path = os.path.join(target_path,"LOINC_1.6_Top2000CommonLabResultsUS.pdf")
    docs = []
    loinc_pattern = re.compile(r"^(\d{4,5}-\d)\s+(.*)")
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages[2:], start=3):
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) <= 3:
                    continue
                for row in table[1:]:
                    if not row[1]:
                        continue
                    text = row[1].strip()
                    match = loinc_pattern.match(text)
                    loinc_code = None
                    if match:
                        loinc_code = match.group(1)
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "loinc": loinc_code,
                                "page": page_num,
                                "rank": row[0]}))
        print("Docs created:", len(docs))

    # # Create Chroma vector store
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client)
    print(f"Stored {len(docs)} rows")
    print(len(docs))

if __name__=="__main__":
    ingest_table()