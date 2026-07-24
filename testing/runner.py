import time
from pathlib import Path

from pathlib import Path

import sys
# ROOT = Path(__file__).resolve().parent.parent       #Eov-pulse directory path
# sys.path.insert(0, str(ROOT))       #sets the root directory to the system path so that we can import modules from it

from testing.clinical import extract_tables_and_text, run_clinical_analysis
from testing.models import TestResult


TEST_FOLDER = "test_documents"


def run_all_tests():

    results = []

    pdfs = sorted(Path(TEST_FOLDER).glob("*.pdf"))

    for pdf in pdfs:
        print(pdf.name)
        start = time.time()

        try:

            extracted_text = extract_tables_and_text(str(pdf))
            
            response = run_clinical_analysis(extracted_text)

            execution_time = round(time.time() - start, 2)

            results.append(
                TestResult(
                    filename=pdf.name,
                    status="PASS",
                    execution_time=execution_time,
                    # extracted_text=extracted_text,
                    extracted_data=response.get("extracted_data", ""),
                    guideline_context=response.get("guideline_context", ""),
                    final_plan=response.get("final_plan", "")
                )
            )

        except Exception as e:

            execution_time = round(time.time() - start, 2)

            results.append(
                TestResult(
                    filename=pdf.name,
                    status="FAIL",
                    execution_time=execution_time,
                    error=str(e)
                )
            )

    return results