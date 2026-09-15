import json
import pandas as pd
import streamlit as st
from pathlib import Path

# Gets the folder where your current script resides
SCRIPT_DIR = Path(__file__).resolve().parent

# Automatically builds the absolute path to your folder
LOG_FILE = SCRIPT_DIR / "ingest_logs" / "ingest_log.jsonl"

print(f"Python is looking exactly here: {LOG_FILE}")

print(LOG_FILE)
st.set_page_config(page_title="RAG Ingest Dashboard", layout="wide")

st.title("📊 RAG Ingestion Dashboard")

if not LOG_FILE.exists():
    st.warning("No ingestion logs found.")
    st.stop()

records = []

with open(LOG_FILE) as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

st.metric("Total Jobs", len(df))
st.metric("Successful", (df["status"] == "SUCCESS").sum())
st.metric("Failed", (df["status"] == "FAILED").sum())

st.divider()

st.dataframe(df, use_container_width=True)

st.divider()

failed = df[df.status == "FAILED"]

if len(failed):

    st.subheader("Failed Jobs")

    for _, row in failed.iterrows():

        st.error(f"""
Document : {row.document_name}

Time : {row.timestamp}

Error : {row.error}
""")

st.divider()

st.subheader("Latest JSON Log")

st.json(records[-1])