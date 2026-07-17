import json
import os
from datetime import datetime
from pathlib import Path

# Gets the folder where your current script resides
SCRIPT_DIR = Path(__file__).resolve().parent

# Automatically builds the absolute path to your folder
LOG_FILE = SCRIPT_DIR / "ingest_logs" / "ingest_log.jsonl"

os.makedirs(SCRIPT_DIR / "ingest_logs", exist_ok=True)


def log_ingestion(
    document_name,
    status,
    rows_processed,
    collection_name,
    error=None,
):
    record = {
        "document_name": document_name,
        "collection_name": collection_name,
        "rows_processed": rows_processed,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "error": error,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")