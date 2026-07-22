from rapidfuzz import fuzz
UNIT_CONVERSION = {
    # Hemoglobin
    "g/dl": ("g/L", 10),
    "gm%": ("g/L", 10),
    "g/l": ("g/L", 1),

    # WBC
    "/ul": ("10^9/L", 0.001),
    "/cumm": ("10^9/L", 0.001),
    "10^3/ul": ("10^9/L", 1),
    "10^9/l": ("10^9/L", 1),

    # RBC
    "m/cumm": ("10^12/L", 1),
    "million/ul": ("10^12/L", 1),
    "10^6/ul": ("10^12/L", 1),
    "10^12/l": ("10^12/L", 1),

    # Platelets
    "lakhs/cumm": ("10^9/L", 100),
    "10^3/ul": ("10^9/L", 1),

    # Creatinine
    "mg/dl": ("µmol/L", 88.4),
    "umol/l": ("µmol/L", 1),

    # Glucose
    "mg/dl": ("mmol/L", 0.0555),
    "mmol/l": ("mmol/L", 1)
}

def normalize_unit(unit):
    unit = unit.lower().strip()

    mapping = {
        # "gm%": "g/dl",
        "gm/dl": "g/dl",
        "gm": "g",
        "g/l": "g/l",
        "/cumm": "/cumm",
        "/cu.mm": "/cumm",
        "/mm3": "/cumm",
        "cells/ul": "/ul",
        "µl": "/ul",
        "/µl": "/ul",
        "ul": "/ul",
        "10³/ul": "10^3/ul",
        "10^3/uL": "10^3/ul",
        "10³/uL": "10^3/ul",
        "/uL": "/ul",
        "/UL": "/ul",
        "gm%": "gm%",
        "g/dL": "g/dl",
        "g/L": "g/l",
        "µmol/L": "µmol/l"
    }

    return mapping.get(unit, unit)

def convert(value, unit):
    print(type(value))
    unit = normalize_unit(unit)
    value=float(value)
    print(type(value))
    if unit not in UNIT_CONVERSION:
        return value, unit

    base_unit, factor = UNIT_CONVERSION[unit]

    return value * factor, base_unit

def update_unit(content,lab):
    # Parse the string
    parts = {}

    for item in content.split(", "):
        key, value = item.split(": ", 1)
        parts[key] = value

    # Normalize if possible
    if {"lower_limit", "upper_limit", "unit"} <= parts.keys():

        unit = normalize_unit(parts["unit"])
        print("normalized unit")
        if unit in UNIT_CONVERSION:

            base_unit, factor = UNIT_CONVERSION[unit]
            print("converting unit and limits")
            parts["lower_limit"] = str(
                float(parts["lower_limit"]) * factor
            )

            parts["upper_limit"] = str(
                float(parts["upper_limit"]) * factor
            )

            
            parts["unit"] = base_unit
            print("done")
            # Convert critical limits if present
            if "critical_low" in parts:
                parts["critical_low"] = str(
                    float(parts["critical_low"]) * factor
                )

            if "critical_high" in parts:
                parts["critical_high"] = str(
                    float(parts["critical_high"]) * factor
                )
            print(f"biomarker name : {parts["biomarker_name"]} and lab name: {lab["lab_name"]}")
            token_ratio = fuzz.token_sort_ratio(parts["biomarker_name"],lab["lab_name"])
            print(token_ratio)
            if token_ratio>90:
                if float(parts["lower_limit"])<=lab["value"] and float(parts["upper_limit"])>=lab["value"]:
                    parts["status"]="normal"
                elif float(parts["lower_limit"])>lab["value"]:
                    parts["status"]="low"
                elif float(parts["upper_limit"])<lab["value"]:
                    parts["status"]="high"
                else:
                    parts["status"]="unknown"
                print(parts["status"])

        else:
            parts["status"]="unknown"
    return ", ".join(f"{k}: {v}" for k, v in parts.items())