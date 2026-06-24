import os
import re
import chromadb
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Note: Use localhost here if running from your terminal, 
# but CHROMA_HOST if running inside the same Docker network.
print("Starting data ingestion...")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def run_ingestion():
    target_path = os.path.join("..", "data")
    print(f"Loading PDF files from {target_path}...")
    loader = DirectoryLoader(target_path, glob="*.pdf", loader_cls=PyPDFLoader)
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

    # High-value clinical and biochemical keywords
    MEDICAL_KEYWORDS = {
    "history", "chief complaint", "symptoms", "diagnosis",
    "prognosis", "assessment", "impression", "findings",
    "clinical", "medical history", "family history",

    # Vital signs
    "blood pressure", "heart rate", "pulse", "temperature",
    "respiratory rate", "oxygen saturation", "spo2",

    # Laboratory tests
    "hemoglobin", "hb", "hematocrit", "wbc", "rbc",
    "platelet", "glucose", "hba1c", "creatinine",
    "urea", "bun", "sodium", "potassium", "chloride",
    "calcium", "bilirubin", "albumin", "cholesterol",
    "triglycerides", "ldl", "hdl", "esr", "crp",

    # Common diseases/conditions
    "diabetes", "hypertension", "anemia", "asthma",
    "copd", "heart disease", "stroke", "infection",
    "cancer", "tumor", "malignancy", "pneumonia",
    "covid", "arthritis", "kidney disease",
    "liver disease", "thyroid", "obesity",

    # Imaging
    "x-ray", "ct", "mri", "ultrasound",
    "scan", "radiology", "lesion", "mass",
    "nodule", "fracture", "effusion",

    # Procedures
    "biopsy", "surgery", "operation",
    "catheterization", "intubation",
    "transplant", "dialysis",

    # Medications
    "medication", "prescription", "dose",
    "tablet", "capsule", "injection",
    "antibiotic", "analgesic", "insulin",

    # Report terminology
    "normal", "abnormal", "positive", "negative",
    "elevated", "decreased", "mild", "moderate",
    "severe", "acute", "chronic", "stable",
    "critical", "follow-up", "recommendation",

    # Blood counts
    "neutrophils", "lymphocytes", "monocytes",
    "eosinophils", "basophils",

    # Liver function
    "alt", "ast", "alp", "ggt",

    # Kidney function
    "egfr", "proteinuria", "albuminuria",

    # Cardiology
    "ecg", "ekg", "echocardiogram",
    "troponin", "arrhythmia", "ischemia",

    # Urine analysis
    "urinalysis", "protein", "ketones",
    "blood", "leukocytes", "nitrite"
    }

    # Clues that a chunk is just boilerplate, a bibliography, or a index
    JUNK_PATTERNS = [
    r"(?i)isbn\s\d+",                     # Copyright info
    r"(?i)published\sby",                 # Publisher boilerplate
    r"(?i)pp\.\s\d+–\d+",                 # Citation page ranges
    r"(?i)doi:\s10\.\S+",                 # Digital Object Identifiers (safe edge termination)
    r"(?i)contents\s+\d+",                # Table of contents lines
    r"^\s*\d+\s*$",                       # Lines that are just page numbers
    
    # --- FIXED FOR MEDICAL VALUES & TABLES ---
    
    # Strictly matches HTML tags (<br>, <div>). Ignores mathematical comparisons (< 180, P < 0.001)
    r"</?[a-zA-Z][^>]*>",                       
    
    # Matches brackets containing ONLY specific metadata words (e.g. [Confidential], [Page 1]). 
    # Leaves medical ranges like [(180–300 pg/mL)] completely untouched.
    r"\[\s*(?:page|pg|confidential|section|sidebar)\b[^\]]*\]",
    
    # Matches repeated punctuation but avoids stripping spaces or table cell separators.
    # Protects hyphenated ranges like "18–24 years" by ensuring it only targets 3+ repetitions.
    r"[\-_*=]{3,}",                            
    
    # Matches consecutive dots (like text...text) but preserves table row leading dots (minimum 4 dots).
    r"\.{4,}",
    
    # Matches page headers but excludes 'pg' to prevent deleting 'pg/mL' (picograms)
    r"(?i)\b(page|section|sect\.)\.?\s*\d+|\bconfidential\b", 
    
    # Matches system timestamps
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",    
    
    # Matches URLs and file paths
    r"https?://\S+|www\.\S+|[a-zA-Z]:\\[^\s]+",  
    
    # Relaxes OCR noise filter. No longer deletes standalone math symbols or structural punctuation.
    r"\b[^\w\s]{3,}\b",                          
    
    # Matches excessive whitespace but leaves single newlines/tabs intact to preserve table columns
    r"[ \t]{2,}"                               
]


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
        if len(words_found) >= 1 and alpha_ratio > 0.30:
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
        collection_name="local_rag"
    )
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()
