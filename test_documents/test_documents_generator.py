import fitz  # PyMuPDF
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ==========================
# Configuration
# ==========================
INPUT_FOLDER = ROOT /"documents_to_split"
OUTPUT_FOLDER = ROOT / "test_documents"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================
# Read PDF
# ==========================
pdf_files = list(INPUT_FOLDER.glob("*.pdf"))

# Print your files to verify
for pdf_path in pdf_files:
    doc = fitz.open(pdf_path)

    rows = []

    for page in doc:
        # Extract words with their coordinates
        words = page.get_text("words")

        # Sort by vertical position, then horizontal position
        words.sort(key=lambda w: (round(w[1]), w[0]))

        current_y = None
        current_row = []

        for word in words:
            x0, y0, x1, y1, text, *_ = word

            y = round(y0)

            if current_y is None:
                current_y = y

            # New row if Y changes significantly
            if abs(y - current_y) > 3:
                if current_row:
                    rows.append(" ".join(current_row))
                current_row = [text]
                current_y = y
            else:
                current_row.append(text)

        if current_row:
            rows.append(" ".join(current_row))

    doc.close()

print(f"Found {len(rows)} rows.")

# ==========================
# Create one PDF per row
# ==========================
for i, row in enumerate(rows, start=1):
    filename = os.path.join(OUTPUT_FOLDER, f"row_{i:04d}.pdf")

    c = canvas.Canvas(filename, pagesize=letter)

    width, height = letter

    text_obj = c.beginText()
    text_obj.setTextOrigin(50, height - 50)
    text_obj.setFont("Helvetica", 12)

    # Wrap long lines
    max_chars = 90
    for j in range(0, len(row), max_chars):
        text_obj.textLine(row[j:j+max_chars])

    c.drawText(text_obj)
    c.save()

print("Done!")