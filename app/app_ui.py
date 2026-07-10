import streamlit as st
from pypdf import PdfReader
import pdfplumber
from streamlit_geolocation import streamlit_geolocation
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import urllib.parse
import re
from graph import clinical_agent  # Importing your working graph
from map_scraper import fetch_local_doctors, extract_specialists
from gmap_test import gmap

# doctors = [
#     {
#     "name": "Oliva Skin, Hair & Laser Clinic Shivaji Nagar, Pune: Laser Hair Removal, Acne Scar, PRP, Skin Whitening Treatments",
#     "category": "Dermatologist",
#     "phone": "+91 89777 55434",
#     "address": "Level 1, Deccan 99 Mall, No 1258, Jangali Maharaj Rd, opposite Deccan Avenue, Pulachi Wadi, Shivajinagar, Pune, Maharashtra 411004, India",
#     "latitude": 18.517222999999998,
#     "longitude": 73.84366399999999,
#     "rating": "4.9 stars",
#     "reviews_count": 1314,
#     "website": "https://locations.olivaclinic.com/oliva-clinic/pune/shivaji-nagar/oliva-skin-hair-and-body-clinic-in-shivaji-nagar-pune--mSY32C/home",
#     "open_state": "Unknown"
#   },
#   {
#     "name": "Clear Skin",
#     "category": "Dermatologist",
#     "phone": "+91 95845 84111",
#     "address": "CTC NO -94, 16 F.P NO -38/16, Prabhat Rd, Erandwane, Pune, Maharashtra 411004, India",
#     "latitude": 18.5142842,
#     "longitude": 73.8342235,
#     "rating": "4.7 stars",
#     "reviews_count": 1508,
#     "website": "https://www.clearskin.in/best-skin-care-clinic-prabhat-road-pune/",
#     "open_state": "Unknown"
#   },
#   {
#     "name": "Kaya Clinic",
#     "category": "Dermatologist",
#     "phone": "+91 86575 69427",
#     "address": "Ground floor, Mantri Vertex, Law College Rd, opposite Nirmitee Furniture, Murlidhar Smruti Society, Apex Colony, Erandwane, Pune, Maharashtra 411004, India",
#     "latitude": 18.5101427,
#     "longitude": 73.8301436,
#     "rating": "4.8 stars",
#     "reviews_count": 845,
#     "website": "https://clinics.kaya.in/near-me/pune/Law-College-Road/kaya-skin-hair-clinic-in-Law-College-Road-pune--1pvSMu/home",
#     "open_state": "Unknown"
#   },
#   {
#     "name": "Taj Skin Hair Laser Clinic Dermatologist Kothrud Pune",
#     "category": "Dermatologist",
#     "phone": "+91 77969 69797",
#     "address": "Stilt floor, Vishnu Arcade, Karve Rd, next to Hotel Sheetal, near Karve statue, Mayur Colony, Kothrud, Pune, Maharashtra 411038, India",
#     "latitude": 18.5024718,
#     "longitude": 73.8162127,
#     "rating": "4.9 stars",
#     "reviews_count": 713,
#     "website": "https://www.tajskin.in/",
#     "open_state": "Unknown"
#   },
#   {
#     "name": "Asia Institute of Hair Transplant",
#     "category": "Skin care clinic",
#     "phone": "+91 72763 71007",
#     "address": "1st Floor, Nandan Pride, Karve Rd, left to Karve Putala, Mayur Colony, Kothrud, Pune, Maharashtra 411038, India",
#     "latitude": 18.5025326,
#     "longitude": 73.81555019999999,
#     "rating": "4.9 stars",
#     "reviews_count": 504,
#     "website": "https://www.skinhairsurgery.com/",
#     "open_state": "Unknown"
#   }
# ]




print("Starting Streamlit UI...")
# 1. Page Configuration (Keep original UI)
st.set_page_config(page_title="EOV Pulse", layout="wide")

st.title("🩺 Clinical Decision & Support System")
st.markdown("---")

# 2. Sidebar for status (Keep original UI)
with st.sidebar:
    st.header("System Status")
    st.success("Ollama: Connected")
    st.success("ChromaDB: Connected")
    st.info("Model: Llama 3 (Reasoning)")
print("UI components set up. Ready for file upload and agent invocation.")
# 3. Helper function to "Freeze" AI results
# This prevents rerunning the agent when you click the button
@st.cache_data
def run_clinical_analysis(text):
    print("Running clinical analysis agent...")
    inputs = {
        "report_text": text,
        "extracted_data": "",
        "guideline_context": "",
        "final_plan": ""
    }
    print("inputs prepared for agent:")
    return clinical_agent.invoke(inputs)

# 4. Session State for clearing the uploader
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def clear_report():
    st.session_state["uploader_key"] += 1
    # This resets the file uploader and clears the cache for the next file
    st.cache_data.clear()

# 5. File Upload (Dynamic Key to allow clearing)
uploaded_file = st.file_uploader(
    "Upload Patient Lab Report (PDF)", 
    type="pdf", 
    key=f"pdf_uploader_{st.session_state['uploader_key']}"
)

######################################### getting user location from browser ##############################
location = streamlit_geolocation()
addr=""
coordinates=[]
# 2. Extract coordinates if the user grants permission
if location and location.get("latitude") and location.get("longitude"):
    lat = location["latitude"]
    lng = location["longitude"]
    coordinates.append(lat)
    coordinates.append(lng)

    try:
        # Always declare a unique user_agent name per OpenStreetMap usage policy
        geolocator = Nominatim(user_agent="clinical_referral_locator_app")
        geo_response = geolocator.reverse(f"{lat}, {lng}", timeout=10)     
        if geo_response:
            addr=f"{geo_response.address}"
            print(f"\n**Street Address:**\n{addr}")
        else:
            print("Coordinates received, but couldn't resolve a structural address.")
                
    except Exception as e:
        st.error(f"Geocoding service unavailable: {e}")

elif location == {}:
    st.info("💡 Please click the location button above to fetch coordinates.")
else:
    st.warning("⚠️ Location access denied or unavailable. Please enable browser location permissions.")


def extract_tables_and_text(pdf_path):
    complete_text=""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            complete_text+="\n\nnewwpagee\n\n"
            
            # Extract plain text with layout preserved
            text = page.extract_text(layout=True)
            complete_text+=f"\n{text}"
            
            # Extract structured tables (e.g., Vitamin D, WBC counts)
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    complete_text+=f"\n{row}"  # Output as a list of strings representing cells
    return complete_text


if uploaded_file is not None:
    with st.spinner("Processing report and consulting WHO guidelines..."):
        print("✅ PDF uploaded successfully!")
        # A. Extract text from PDF
        
        raw_text=extract_tables_and_text(uploaded_file)

        # reader = PdfReader(uploaded_file)
        print("Extracting text from PDF...")
        # raw_text = "\n\nnewwpagee\n\n".join([page.extract_text() for page in reader.pages])
        print(raw_text[:50] + "...")  # Show a preview of the extracted text
        # B. Run the Agent (Cached version)
        print("Invoking the clinical analysis agent...")
        result = run_clinical_analysis(raw_text)
        print("✅ Clinical analysis complete!")

        #extracting specialists from llm response
        specialists=extract_specialists(result["final_plan"])
        #fetching nearby specialists
        nearby_specialists=fetch_local_doctors(specialists,addr,coordinates)

        # C. Display Results in Columns (Keep original UI)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Extracted Lab Values")
            st.info(result["extracted_data"])
            print("Extracted lab values:")
            with st.expander("View Referenced WHO Guidelines"):
                print("Referenced WHO Guidelines:")
                st.write(f"{result["guideline_context"][:2000]}...")

        with col2:
            st.subheader("📝 Drafted Clinical Action Plan")
            st.success(result["final_plan"])
            print("Drafted clinical action plan:")

            # 6. Functional "Approve" Button
            if st.button("Approve & Sign Referral"):
                st.balloons()
                st.success("✅ Referral signed and saved to Patient History!")
                
                # Option to clear the screen and start over
                st.button("Reset / New Patient", on_click=clear_report)
        with st.spinner("Locating nearby Doctors on map"):
            gmap(nearby_specialists)