import json
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb
import os
from app.llm.manager import invoke_auto,invoke
import asyncio
from json_repair import repair_json
load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_client = Chroma(client=client, collection_name="local_rag3", embedding_function=embeddings)

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
1. Extract exactly 5 key laboratory and clinically relevant findings from the entire report.
2. Generate a concise summary (50-100 words).
3. Avoid missing important parameters

Rules:
- Analyze the complete report, including laboratory results, clinical notes, impressions, and recommendations.
- Do not invent laboratory values, reference ranges, or diagnoses.
- Refer official guidelines.
- Base the summary only on information present in the report.

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
            "summary": "Narrative paragraph describing the report...",
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
                    "reference_label": "< 200",            # display string
                    "status": "elevated"                   # elevated | low | in_range  
                }},
            ]
        }}
    ]
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
        print("type of labs is:::::::::: ",type(labs))
    except Exception as e:
        raise Exception(
            f"Invalid JSON returned by Gemini\n{e}"
            f"Response:\n{text}"
        )
    print("\nparameters are ::::::::::::::::", labs.get("extracted_data")[0].get("parameters"))
    print("\n extracted data is ::::::::::::::::", labs.get("extracted_data")[0])
    return {
        "extracted_data": labs.get("extracted_data"),
        # "guideline_context":labs.get("guideline_context"),
        # "final_plan":labs.get("final_plan"),
    }

def retrieve_guidelines_node(state: AgentState):
    """
    Retrieve guideline documents for each extracted laboratory test.

    Returns:
        guideline_context -> list[str]
    """
    labs = state["extracted_data"]
    #Out Of Range (OOR) Parameters to verify
    OOR_test=[lab for lab in labs[0].get("parameters",[]) if lab.get("status")!="in_range"]
    # try:
    #     raw_guidelines = labs[0].get("reference_guidelines")

    #     # Normalize to list
    #     if raw_guidelines is None:
    #         raw_guidelines = []
    #     elif isinstance(raw_guidelines, str):
    #         raw_guidelines = [raw_guidelines]

    # except (IndexError, AttributeError):
    #     raw_guidelines = []
    print("inside guidelines function")
    chroma_guidelines=[]
    chroma_retriever = chroma_client.as_retriever(search_kwargs={"k": 2})
    print("before retrieval")
    for lab in OOR_test:
        query=f"{lab.get("name")},{lab.get("unit")}"
        docs=chroma_retriever.invoke(query)
        chroma_guidelines.extend(docs)
    print("after retrieval")
    guideline_context = [
        item.page_content.strip()
        for item in chroma_guidelines
    ]
    if not guideline_context:
        print("No valid reference guidelines found.")
    else:
        print(f"Loaded {len(guideline_context)} valid guidelines.")
        # print(guideline_context)
        for i,item in enumerate(guideline_context):
            print(f"\n\nguideline content {i+1} is {item}\n\n")

    # print("guideline context :", guideline_context)
    return {
        "guideline_context": guideline_context
    }

def clinical_reasoning_node(state: AgentState):
    """
    Final Gemini reasoning.

    Input:
        extracted_data
        guideline_context

    Output:
        final_plan
    """
    print("in reasoning function")
    labs = state["extracted_data"]
        #Out Of Range (OOR) Parameters to verify
    OOR_test=[lab for lab in labs[0].get("parameters",[]) if lab.get("status")!="in_range"]

    summary=labs[0].get("summary")
    guidelines=state["guideline_context"]
    prompt = f"""
        You are an expert clinical decision support assistant.

        Your task is to review the previously extracted clinical summary together with the abnormal laboratory parameters and verify them against established clinical practice guidelines (e.g., ADA, KDIGO, ACC/AHA, NCEP ATP IV, ATA, AACE, WHO, ESC, NICE, or other relevant evidence-based guidelines) and provided retrieved guidelines.

        INPUT

        Previous Summary:
        {summary}

        Out-of-Range Parameters:
        {OOR_test}

        Retrieved Guidelines:
        {guidelines}

        TASKS

        1. Review the provided summary.
        2. Verify whether the abnormal laboratory values are clinically significant according to the appropriate official guideline(s) consider retrieved guideline ranges over ranges provided in report and mention it.
        3. Update the summary only if guideline-based interpretation changes or improves it (example: official ranges differ from ranges provided in report and give both conflicting ranges).
        4. Generate concise guideline context for every relevant abnormal finding.
        5. Produce a prioritized clinical action plan.
        6. Recommend the single most appropriate medical specialist.
        7. Suggest appropriate follow-up investigations or repeat testing only when supported by guidelines or standard clinical practice.

        RULES

        - Base recommendations only on the provided summary and abnormal parameters.
        - Do NOT invent laboratory values.
        - Do NOT invent diagnoses.
        - Do NOT modify reported numerical values.
        - Cite only relevant, widely accepted clinical guidelines.
        - If multiple guidelines apply, include all relevant ones.
        - consider retrieved guidelines.
        - Keep guideline descriptions concise.
        - Return ONLY valid JSON.
        - If the report does not contain enough information for a recommendation, state this in the follow-up plan.

        Before responding, verify that:
        1. The output is valid JSON.
        2. Every property name is enclosed in double quotes.
        3. There are no trailing commas.
        4. The JSON can be parsed by Python's json.loads().

        Return only the validated JSON object.
        {{
        "summary": "...",   #updated summary if any change
        "guideline_context": "...",   # long text, OR list of {{"title":..., "description":...}}
        "final_plan": {{
            "specialist": "...", #most appropriate speacialists
            "follow_up_plan": "...",  # kept for backward-compat / plain-text fallback
            "actions": [
                {{
                    "priority": "high",  # high | medium | monitor
                    "parameter_label": "LDL 151.92 mg/dL, non-HDL 183.3 mg/dL,
                    "title": "Bring LDL and non-HDL cholesterol down",
                    "description": "LDL sits 52 mg/dL above the ATP IV target...",
                    "steps": ["Reduce saturated fat to under 7% of daily calories...", "..."],
                    "retest_window": "Lipid panel in 12 weeks",
                    "refer_to": "Cardiology or internal medicine"
                }},
            ],
            "retest_schedule": [
                {{"when": "2\u20134 weeks", "detail": "Repeat total testosterone, morning fasting sample..."}},
            ],
        }},
        "next_retest": "In 2\u20134 weeks",
        "next_retest_detail": "Repeat testosterone on a morning sample"}},
        "flagged_systems": [
            {{"name": "Cardiovascular / lipids", "detail": "4 out of range", "status": "bad"}},
            {{"name": "Glycaemic control", "detail": "Normal", "status": "good"}},
        ]
    """
    response = invoke(prompt)
    # print(response)
    print("after invoking")
    if isinstance(response, list):
            print("inside if block")
            text1=[]
            for block in response:
                if isinstance(block, dict) and block.get("type")=='text':
                    text1.append(block["text"])
    
            text = "".join(text1)
            print("text in reasoning is::::::::::::::::", text)
    else:
        print("inside else block")
        text = response
    # print("type of text is::::::::::::::::::::::::::::::", type(text))
    text = text.strip()
    if text.startswith("```"):
        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
    
    try:
        print(f"\ntype of text is:::::::::::{type(text)}")
        reasoning = repair_json(text, return_objects=True)
        # reasoning = json.loads(text)
        print("type of labs is:::::::::: ",type(reasoning))
    except Exception as e:
        raise Exception(
            f"Invalid JSON returned by Gemini\n{e}\n"
            f"Response:\n{text}"
        )
    labs[0]["summary"]=reasoning.get("summary")
    labs[0]["next_retest"]=reasoning.get("next_retest")
    labs[0]["next_retest_detail"]=reasoning.get("next_retest_detail")
    labs[0]["flagged_systems"]=reasoning.get("flagged_systems")
    return {
            "extracted_data": labs,
            "guideline_context":reasoning.get("guideline_context"),
            "final_plan":reasoning.get("final_plan"),
        }
    
# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("extractor", extract_labs_node)
workflow.add_node("researcher", retrieve_guidelines_node)
workflow.add_node("writer", clinical_reasoning_node)

# Define Edges (The flow)
workflow.set_entry_point("extractor")
# workflow.add_edge("extractor", END)

workflow.add_edge("extractor", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

clinical_agent = workflow.compile()