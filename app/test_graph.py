from graph import clinical_agent

def run_test():
    # 1. Simulate the "Messy" text that would come from a PDF parser
    messy_report_text = """
    PATIENT RECORD #99283
    Date: 2024-05-12
    Patient shows signs of fatigue and increased thirst.
    Lab Results:
    - HbA1c: 8.5% (High)
    - Fasting Glucose: 145 mg/dL
    - LDL: 120
    - BP: 130/85
    Notes: Patient has family history of Type 2 Diabetes. 
    Review clinical guidelines for next steps.
    """

    print("--- Starting Clinical Assistant Agent ---")
    
    # 2. Prepare the initial state
    initial_state = {
        "report_text": messy_report_text,
        "extracted_data": "",
        "guideline_context": "",
        "final_plan": ""
    }

    # 3. Invoke the Graph
    try:
        final_output = clinical_agent.invoke(initial_state)

        print("\n[STEP 1: EXTRACTED DATA]")
        print(final_output["extracted_data"])

        print("\n[STEP 2: GUIDELINE CONTEXT]")
        print(final_output["guideline_context"])

        print("\n[STEP 3: FINAL CLINICAL PLAN]")
        print(final_output["final_plan"])
        
    except Exception as e:
        print(f"Error running agent: {e}")
        print("Check if your Ollama container is running at http://docker.internal")

if __name__ == "__main__":
    run_test()
