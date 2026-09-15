import csv
import io
from pathlib import Path

# ==========================================================
# Paste your raw data below
# ==========================================================
raw_medical_data = """"""


# ==========================================================
# Output CSV
# ==========================================================
ROOT = Path(__file__).resolve().parent.parent.parent

output_filename = ROOT / "data" / "thresholds_rag2.csv"

header = [
    "biomarker_id",
    "panel_name",
    "normalized_name",
    "biomarker_name",
    "sample_type",
    "demographic_group",
    "lower_limit",
    "upper_limit",
    "unit",
    "critical_low",
    "critical_high",
    "notes_and_context",
]

# ==========================================================
# Parse raw text
# ==========================================================

rows = []

reader = csv.reader(io.StringIO(raw_medical_data))

for row in reader:

    # Skip blank lines
    if not row:
        continue

    if len(row) == 0:
        continue

    # Remove surrounding whitespace
    row = [cell.strip() for cell in row]

    # Skip empty rows
    if all(cell == "" for cell in row):
        continue

    # Ensure exactly 12 columns
    if len(row) < 12:
        row.extend([""] * (12 - len(row)))

    elif len(row) > 12:
        row = row[:11] + [",".join(row[11:])]

    rows.append(row)

# ==========================================================
# Write CSV
# ==========================================================

with open(output_filename, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(
        f,
        quoting=csv.QUOTE_MINIMAL
    )

    writer.writerow(header)
    writer.writerows(rows)

print(f"Successfully wrote {len(rows)} rows to '{output_filename}'")