from fastapi import FastAPI
import sys
from pathlib import Path
from dataclasses import asdict
import csv
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "test_results" / "analysis_results.csv"      #csv file to save the output test results

from testing.runner import run_all_tests

app = FastAPI(title="Clinical AI Test API")

#function to save results in csv file
def save_results(results):

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "FileName",
            "Processing Date & Time",
            "extracted data",
            "guidelines context",
            "final plan / error"
        ])

        for r in results:

            writer.writerow([
                r.filename,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                r.extracted_data if r.status == "PASS" else "",
                r.guideline_context if r.status == "PASS" else "",
                r.final_plan if r.status == "PASS" else r.error
            ])


@app.get("/")
def home():
    return {
        "message": "Clinical AI Testing Service"
    }


@app.get("/run-tests")
def run_tests():

    results = run_all_tests()
    save_results(results)
    return {
        "total": len(results),
        "passed": sum(r.status == "PASS" for r in results),
        "failed": sum(r.status == "FAIL" for r in results),
        "results": [asdict(r) for r in results]
    }