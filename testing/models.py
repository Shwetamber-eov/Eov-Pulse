from dataclasses import dataclass


@dataclass
class TestResult:
    filename: str
    status: str
    execution_time: float
    extracted_text: str = ""
    extracted_data: str = ""
    guideline_context: str = ""
    final_plan: str = ""
    error: str = ""