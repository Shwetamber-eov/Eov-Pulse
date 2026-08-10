import json
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from app.llm.manager import invoke_auto
import asyncio

load_dotenv()

# 1. Define the State
class AgentState(TypedDict):
    report_text: list
    extracted_data: list
    guideline_context: list
    final_plan: list

# --- NODES ---
def extract_labs_node(state: AgentState):

    """
    Gemini extracts all laboratory values from the report.

    Output:
        extracted_data -> Python list
    """

    report = state["report_text"]

    prompt = f"""
You are a clinical decision support assistant.

Analyze the entire clinical report below.

Tasks:
1. Extract exactly 5 key laboratory and clinically relevant findings from the entire report .
2. Recommend the single most appropriate medical specialist, if indicated.
3. Generate a concise follow-up plan (50-100 words).

Rules:
- Analyze the complete report, including laboratory results, clinical notes, impressions, and recommendations.
- Do not invent laboratory values, reference ranges, or diagnoses.
- Refer official guidelines.
- Base the summary, specialist recommendation, and follow-up plan only on information present in the report.
- If the report does not contain enough information for a recommendation, state this in the follow-up plan.

Before responding, verify that:
1. The output is valid JSON.
2. Every property name is enclosed in double quotes.
3. There are no trailing commas.
4. The JSON can be parsed by Python's json.loads().

Return only the validated JSON object.

Schema:
{{
    "extracted_data": [
        {{
            "summary": "Narrative paragraph describing the report."(around 50 words),
            "patient": {{"age": 39, "sex": "male"}},
            "report_date": "28 July 2026",
            "parameters": [
                {{
                    "name": "Total cholesterol",
                    "category": "Lipid profile",          # groups table rows
                    "guideline_label": "NCEP ATP IV",      # small tag per group
                    "value": 235.6,
                    "unit": "mg/dL",
                    "reference_low": None,
                    "reference_high": 200,
                    "reference_label": "< 200",            
                    "status": "elevated",                  # elevated | low | in_range
                }},
            ],
            "flagged_systems": [
                {{"name": "Cardiovascular / lipids", "detail": "4 out of range", "status": "bad"}},
                {{"name": "Glycaemic control", "detail": "Normal", "status": "good"}},
            ],
            "next_retest": "In 2-4 weeks",
            "next_retest_detail": "Repeat testosterone on a morning sample",
        }}
    ],
    "guideline_context": "",   # long text, OR list of {{"title":..., "description":...}}
    "final_plan": {{
        "specialist": "",
        "follow_up_plan": "",  
        "actions": [
            {{
                "priority": "high",  # high | medium | monitor
                "parameter_label": "LDL 151.92 mg/dL, non-HDL 183.3 mg/dL, hs-CRP 2.60 mg/L",
                "title": "Bring LDL and non-HDL cholesterol down",
                "description": "LDL sits 52 mg/dL above the ATP IV target.",
                "steps": ["Reduce saturated fat to under 7% of daily calories.", ""],
                "retest_window": "Lipid panel in 12 weeks",
                "refer_to": "Cardiology or internal medicine",
            }},
        ],
        "retest_schedule": [
            {{"when": "2-4 weeks", "detail": "Repeat total testosterone, morning fasting sample."}},
        ],
    }},
}}

"""
    response = invoke_auto(
        pages= report,
        system_prompt=prompt
    )
    
    if isinstance(response, list):
            print("inside if block")
            text1=[]
            for block in response:
                if isinstance(block, dict) and block.get("type")=='text':
                    text1.append(block["text"])

            text = "".join(text1)
    else:
        print("inside else block")
        text = response
    # print("text is:::::::::::::::", text)
    text = text.strip()
    if text.startswith("```"):
        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:
        labs = json.loads(text)

    except Exception as e:
        raise Exception(
            f"Invalid JSON returned by Gemini\n{e}"
        )
    # print(f'\n\n"extracted_data": {labs.get("extracted_data")},\n\n"guideline_context":{labs.get("guideline_context")},\n\n"final_plan":{labs.get("final_plan")}')
    return {
        "extracted_data": labs.get("extracted_data"),
        "guideline_context":labs.get("guideline_context"),
        "final_plan":labs.get("final_plan")
    }

# def retrieve_guidelines_node(state: AgentState):
#     """
#     Retrieve guideline documents for each extracted laboratory test.

#     Returns:
#         guideline_context -> list[str]
#     """
#     labs = state["extracted_data"]

#     try:
#         raw_guidelines = labs[0].get("reference_guidelines")

#         # Normalize to list
#         if raw_guidelines is None:
#             raw_guidelines = []
#         elif isinstance(raw_guidelines, str):
#             raw_guidelines = [raw_guidelines]

#     except (IndexError, AttributeError):
#         raw_guidelines = []

#     guideline_context = [
#         item.strip()
#         for item in raw_guidelines
#         if item and item.strip().lower() not in {
#             "",
#             "none",
#             "n/a",
#             "no action required",
#         }
#     ]

#     if not guideline_context:
#         print("No valid reference guidelines found.")
#     else:
#         print(f"Loaded {len(guideline_context)} valid guidelines.")
#         print(guideline_context)

#     print("guideline context :", guideline_context)
#     return {

#         "guideline_context": guideline_context

#     }

# def clinical_reasoning_node(state: AgentState):
#     """
#     Final Gemini reasoning.

#     Input:
#         extracted_data
#         guideline_context

#     Output:
#         final_plan
#     """

#     extracted_data = state["extracted_data"]
#     print(extracted_data)
#     final_plan=extracted_data[0]
#     print(final_plan)
#     return {"final_plan": final_plan}

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("extractor", extract_labs_node)
# workflow.add_node("researcher", retrieve_guidelines_node)
# workflow.add_node("writer", clinical_reasoning_node)

# Define Edges (The flow)
workflow.set_entry_point("extractor")
# workflow.add_edge("extractor", "researcher")
# workflow.add_edge("researcher", "writer")
# workflow.add_edge("writer", END)
workflow.add_edge("extractor", END)

clinical_agent = workflow.compile()