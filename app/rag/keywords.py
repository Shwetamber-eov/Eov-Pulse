# Clues that a chunk is just boilerplate, a bibliography, or a index
JUNK_PATTERNS = [
r"(?i)isbn\s\d+",                     # Copyright info
r"(?i)published\sby",                 # Publisher boilerplate
r"(?i)pp\.\s\d+–\d+",                 # Citation page ranges
r"(?i)doi:\s10\.\S+",                 # Digital Object Identifiers (safe edge termination)
r"(?i)contents\s+\d+",                # Table of contents lines
r"^\s*\d+\s*$",                       # Lines that are just page numbers
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

# High-value clinical and biochemical keywords
MEDICAL_KEYWORDS = {
    # "history", "chief complaint", "symptoms", "diagnosis",
    # "prognosis", "assessment", "impression", "findings",
    # "clinical", "medical history", "family history",
    "clinical","symptoms","diagnosis","medical history",
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
