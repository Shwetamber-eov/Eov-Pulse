"""
EOV-PULSE — FastAPI backend for the standalone HTML/JS frontend (frontend/index.html)
========================================================================================
This replaces the Streamlit *presentation layer* with plain HTTP endpoints. It reuses
every piece of your existing backend logic unchanged:

    extract_tables_and_text   (testing.clinical)
    clinical_agent.invoke     (app.backend.graph_multiagent_heavy)
    extract_specialists       (app.backend.map_scraper)
    fetch_local_doctors       (app.backend.map_scraper)
    Nominatim reverse geocoding (geopy)

WHERE TO PUT THIS FILE
-----------------------
Place it at the same directory depth your Streamlit page (`streamlit_app.py`, or
whatever it's called) was at, i.e. wherever `ROOT = Path(__file__).resolve().parent
.parent.parent` pointed at your project root in the original file. Adjust the
`.parent` chain below (marked ADJUST ME) if you move it.

RUN IT
------
    pip install fastapi "uvicorn[standard]" python-multipart geopy
    uvicorn backend_api:app --reload --port 8000

Then open http://localhost:8000/ — FastAPI serves frontend/index.html directly, so
the page's fetch('/api/...') calls hit this same server with zero CORS setup.

If you'd rather run the frontend from a different origin (e.g. a separate static
host, or opening the file with `python -m http.server`), CORS is already enabled
below — just set `API_BASE` at the top of index.html's <script> to this server's
URL, e.g. 'http://localhost:8000'.

RESPONSE CONTRACT
-----------------
POST /api/analyze returns:
    {
      "result": { ...whatever clinical_agent.invoke(...) returned... },
      "specialists": [...whatever extract_specialists(...) returned...],
      "nearby_specialists": [
        {
          "name": "...", "specialty": "...", "distance": "...", "address": "...",
          "matched_for": "...", "next_slot": "...",
          "lat": 18.56, "lng": 73.79            # optional — omit to skip the map pin
        }, ...
      ]
    }

The frontend's JS adapters (getNarrative, getParameters, getActions, etc.) read
`result` using the exact same shape documented in the original Streamlit app's
docstring (extracted_data[0].summary/parameters/flagged_systems, guideline_context,
final_plan.actions/retest_schedule). If `clinical_agent.invoke(...)` doesn't return
that shape yet, the page still renders — those sections just show an empty state.

`normalize_specialist_row()` below is where you adapt whatever `fetch_local_doctors()`
actually returns into the flat dict shape the frontend expects. Edit the field
lookups there to match your real data — it's intentionally forgiving (tries a few
common key names) but you know your own schema best.
"""

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# --- ADJUST ME: match the sys.path setup from your original Streamlit file ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# from app.backend.graph_multiagent_heavy import clinical_agent          # noqa: E402
from app.backend.graph_multiagent_heavy_1 import clinical_agent          # noqa: E402
from app.backend.map_scraper import extract_specialists, fetch_local_doctors  # noqa: E402
from testing.clinical import extract_tables_and_text                    # noqa: E402

from geopy.geocoders import Nominatim  # noqa: E402

FRONTEND_DIR = Path(__file__).resolve().parent
geolocator = Nominatim(user_agent="eov_pulse_app")

app = FastAPI(title="EOV-PULSE API")

# Only needed if the frontend is served from a different origin than this API.
# Harmless to leave on even when serving both from here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_clinical_analysis(report_text: str) -> dict:
    """Same call the Streamlit app made — kept as its own function so you can
    reintroduce caching (e.g. functools.lru_cache, or a Redis cache) later."""
    inputs = {
        "report_text": report_text,
        "extracted_data": "",
        "guideline_context": "",
        "final_plan": "",
    }
    return clinical_agent.invoke(inputs)


def normalize_specialist_row(s: dict, index: int) -> dict:
    """Best-effort normalization of whatever fetch_local_doctors() returns into
    the flat shape frontend/index.html expects. Adjust the .get(...) fallbacks
    to match your real field names."""
    lat = s.get("lat") or s.get("latitude") or (s.get("coordinates") or [None, None])[0]
    lng = s.get("lng") or s.get("longitude") or (s.get("coordinates") or [None, None])[1]
    row = {
        "name": s.get("name", "Specialist"),
        "specialty": s.get("specialty", ""),
        "distance": s.get("distance", ""),
        "address": s.get("address", ""),
        "matched_for": s.get("matched_for", ""),
        "next_slot": s.get("next_slot", ""),
    }
    if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
        row["lat"] = lat
        row["lng"] = lng
    return row


# ---------------------------------------------------------------------------
# API routes (must be declared before the catch-all static mount at the bottom)
# ---------------------------------------------------------------------------

@app.post("/api/geocode")
def geocode(lat: float = Form(...), lng: float = Form(...)):
    """Reverse-geocode browser coordinates into a human-readable address,
    used only to rank/label nearby specialists — same as the original
    reverse_geocode() cached function in the Streamlit app."""
    try:
        geo = geolocator.reverse(f"{lat}, {lng}", timeout=10)
        if geo:
            addr = geo.raw.get("address", {})
            if addr:
                suburb= addr.get("suburb") if addr.get("suburb") else ""
                city = addr.get("city") or addr.get("town") or addr.get("village") if addr.get("city") or addr.get("town") or addr.get("village") else ""
                return {"address": f"{suburb},{city}"}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Geocoding service unavailable: {exc}")


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
    address: str = Form(""),
):
    """Full pipeline: PDF -> text -> clinical_agent -> specialists -> nearby doctors.
    Mirrors the original Streamlit flow (extract_tables_and_text ->
    clinical_agent.invoke -> extract_specialists -> fetch_local_doctors) in one call."""
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    contents = await file.read()

    try:
        raw_text = extract_tables_and_text(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read the PDF: {exc}")

    try:
        result = run_clinical_analysis(raw_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    try:
        print("type of result in analyze function is:::::::",type(result))
        specialists = extract_specialists(result.get("final_plan", {}) if isinstance(result, dict) else {})
    except Exception:
        specialists = []

    coordinates = [lat, lng] if lat is not None and lng is not None else []
    try:
        nearby_raw = fetch_local_doctors(specialists, address, coordinates)
    except Exception:
        nearby_raw = []

    nearby = [normalize_specialist_row(s, i) for i, s in enumerate(nearby_raw or []) if isinstance(s, dict)]

    return {
        "result": result,
        "specialists": specialists,
        "nearby_specialists": nearby,
    }


# ---------------------------------------------------------------------------
# Static frontend — declared last so it doesn't shadow the /api/* routes above.
# Serves frontend/index.html at "/" and any other static assets in that folder.
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
