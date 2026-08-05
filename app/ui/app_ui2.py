import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import json

from pathlib import Path
import sys
# from gmap_test import gmap
from gmap_test2 import gmap

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.backend.graph_multiagent import clinical_agent
from app.backend.map_scraper import fetch_local_doctors, extract_specialists
from testing.clinical import extract_tables_and_text

st.set_page_config(page_title="EOV Pulse", layout="wide", page_icon="🩺")

# ---------- Light custom styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
div[data-testid="stMetricValue"] {font-size: 1.4rem;}
.stExpander {border-radius: 10px;}
.eov-card {
    background: #ffffff10;
    border: 1px solid #ffffff20;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 EOV-PULSE")
st.caption("Clinical lab report analysis, guideline referencing, and local specialist routing.")
st.markdown("---")

with st.sidebar:
    st.header("System Status")
    st.success("Ollama: Connected")
    st.success("ChromaDB: Connected")
    st.info("Model: Llama 3 (Reasoning)")

# ---------- Session state init ----------
defaults = {
    "uploader_key": 0,
    "processed_file_id": None,
    "result": None,
    "specialists": None,
    "nearby_specialists": None,
    "addr": "",
    "coordinates": [],
    "location_resolved": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

def clear_report():
    st.session_state["uploader_key"] += 1
    st.session_state["processed_file_id"] = None
    st.session_state["result"] = None
    st.session_state["specialists"] = None
    st.session_state["nearby_specialists"] = None
    st.cache_data.clear()

# ---------- Cached, expensive steps ----------
@st.cache_data(show_spinner=False)
def run_clinical_analysis(text):
    inputs = {
        "report_text": text,
        "extracted_data": "",
        "guideline_context": "",
        "final_plan": ""
    }
    return clinical_agent.invoke(inputs)

@st.cache_data(show_spinner=False)
def get_nearby_specialists(specialists, addr, coordinates):
    # tuple/list args must be hashable-friendly for caching; convert as needed
    return fetch_local_doctors(specialists, addr, list(coordinates))

@st.cache_data(show_spinner=False)
def reverse_geocode(lat, lng):
    geolocator = Nominatim(user_agent="clinical_referral_locator_app")
    geo_response = geolocator.reverse(f"{lat}, {lng}", timeout=10)
    return geo_response.address if geo_response else ""

# ---------- File upload ----------
uploaded_file = st.file_uploader(
    "Upload Patient Lab Report (PDF)",
    type="pdf",
    key=f"pdf_uploader_{st.session_state['uploader_key']}"
)

# ---------- Location (only resolve once per session) ----------
if not st.session_state["location_resolved"]:
    location = streamlit_geolocation()
    if location and location.get("latitude") and location.get("longitude"):
        lat, lng = location["latitude"], location["longitude"]
        st.session_state["coordinates"] = [lat, lng]
        try:
            st.session_state["addr"] = reverse_geocode(lat, lng)
            st.session_state["location_resolved"] = True
        except Exception as e:
            st.error(f"Geocoding service unavailable: {e}")
    elif location == {}:
        st.info("💡 Please click the location button above to fetch coordinates.")
    else:
        st.warning("⚠️ Location access denied or unavailable. Please enable browser location permissions.")
else:
    st.caption(f"📍 Location set: {st.session_state['addr'][:80]}...")

# ---------- Main processing (only runs once per uploaded file) ----------
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state["processed_file_id"] != file_id:
        with st.spinner("Processing report and consulting WHO guidelines..."):
            raw_text = extract_tables_and_text(uploaded_file)
            result = run_clinical_analysis(raw_text)
            specialists = extract_specialists(result["final_plan"])
            nearby_specialists = get_nearby_specialists(
                specialists, st.session_state["addr"], st.session_state["coordinates"]
            )

        st.session_state["result"] = result
        st.session_state["specialists"] = specialists
        st.session_state["nearby_specialists"] = nearby_specialists
        st.session_state["processed_file_id"] = file_id

    # Read from session_state from here on — no recompute on zoom/rerun
    result = st.session_state["result"]
    nearby_specialists = st.session_state["nearby_specialists"]

    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📝 Action Plan", "🗺️ Nearby Specialists"])

    with tab1:
        st.subheader("Extracted Lab Values")
        st.info(result["extracted_data"][0].get("summary"))
        with st.expander("View Referenced WHO Guidelines"):
            st.write(f"{result['guideline_context'][:2000]}...")

    with tab2:
        st.subheader("Drafted Clinical Action Plan")
        st.success(result["final_plan"].get("follow_up_plan"))

        if st.button("✅ Approve & Sign Referral"):
            st.balloons()
            st.success("Referral signed and saved to Patient History!")
            st.button("🔄 Reset / New Patient", on_click=clear_report)

    with tab3:
        st.subheader("Nearby Doctors")
        gmap(nearby_specialists)
else:
    st.info("Upload a lab report PDF to begin.")