import streamlit as st
import folium
from streamlit_folium import st_folium
import urllib.parse
# import pandas as pd
# Sample data matching your ai_ready_doctors structure (with lat/lon added)
ai_ready_doctors = [
    {
        "name": "Oliva Skin, Hair & Laser Clinic Shivaji Nagar, Pune: Laser Hair Removal, Acne Scar, PRP, Skin Whitening Treatments",
        "rating": "4.8 stars",
        "address": "Level 1, Deccan 99 Mall, No 1258, Jangali Maharaj Rd, opposite Deccan Avenue, Pulachi Wadi, Shivajinagar, Pune, Maharashtra 411004, India",
        "latitude": 18.5204,
        "longitude": 73.8567,
    },
    {
        "name": "Clear Skin",
        "rating": "4.5 stars",
        "address": "CTC NO -94, 16 F.P NO -38/16, Prabhat Rd, Erandwane, Pune, Maharashtra 411004, India",
        "latitude": 18.4869,
        "longitude": 73.8405,
    }
]

start_lat = ai_ready_doctors[0]["latitude"]
start_lon = ai_ready_doctors[0]["longitude"]

st.title("Interactive Doctor Map")

# Convert listings to a DataFrame for Streamlit mapping
# df = pd.DataFrame(ai_ready_doctors)

# Split screen into two columns: Map on left, Doctor Details on right
col1, col2 = st.columns([2, 1])

# Ensure session state tracks the clicked doctor index
if "selected_doctor_idx" not in st.session_state:
    st.session_state.selected_doctor_idx = None
if "scroll_trigger" not in st.session_state:
    st.session_state.scroll_trigger = False

with col1:
    st.subheader("Doctor Locations")
    m = folium.Map(location=[start_lat, start_lon], zoom_start=13)
    
    # 1. Loop with enumerate so each marker has a unique index ID
    for idx, doc in enumerate(ai_ready_doctors):
        folium.CircleMarker(
            location=[doc["latitude"], doc["longitude"]],
            radius=8,  # Slightly bigger for easier mobile clicking
            tooltip=doc["name"],
            color="#FF4B4B",
            fill=True,
            fill_color="#FF4B4B",
            fill_opacity=1,
            # CRITICAL: Save the index into the marker options dictionary
            options={"id": idx}  
        ).add_to(m)
    
    # 2. Capture the map interaction payload
    map_data = st_folium(m, width=700, height=500, key="doctor_map")
    
    # 3. Detect if a marker was clicked and update the active index
    if map_data and map_data.get("last_object_clicked"):
        clicked_lat = map_data["last_object_clicked"]["lat"]
        clicked_lng = map_data["last_object_clicked"]["lng"]
        
        # Match back coordinates to find the correct doctor index
        for idx, doc in enumerate(ai_ready_doctors):
            if abs(doc["latitude"] - clicked_lat) < 1e-5 and abs(doc["longitude"] - clicked_lng) < 1e-5:
                if st.session_state.selected_doctor_idx != idx:
                    st.session_state.scroll_trigger = True  # Signal col2 to scroll
                    st.rerun()

with col2:
    st.subheader("Doctor Directory")
    
    # 1. Initialize an empty list to assemble your HTML components
    directory_html_pieces = []
    
    # Open the scrollable container wrapper tag
    directory_html_pieces.append(
        """
        <div id="scrollable-directory" style="height: 500px; overflow-y: scroll; padding-right: 10px; border: 1px solid #ddd; border-radius: 8px;">
        """
    )
    
    # 2. Build out each doctor profile card slice
    for idx, doc in enumerate(ai_ready_doctors):
        is_selected = (st.session_state.selected_doctor_idx == idx)
        
        # Apply the exact same conditional styles safely
        card_border = "border: 2px solid #FF4B4B;" if is_selected else "border: 1px solid #e6e6e6;"
        bg_color = "background-color: #FFF5F5;" if is_selected else "background-color: white;"
        
        query_string = urllib.parse.quote(f"{doc['name']}, {doc['address']}" if doc["address"] else doc["name"])
        gmap_link = f"https://google.com/maps/search/?api=1&query={query_string}"
        
        card_html = f"""
        <div id="doc-card-{idx}" style="{card_border} {bg_color} padding: 15px; margin: 10px 0; border-radius: 6px; font-family: sans-serif;">
            <h3 style="margin-top: 0; color: #333;">{doc['name']}</h3>
            <p style="margin: 5px 0;">⭐ {doc['rating']}</p>
            <p style="margin: 5px 0; color: #666;">📍 {doc['address']}</p>
            <a href="{gmap_link}" target="_blank" style="
                display: inline-block;
                margin-top: 10px;
                padding: 8px 16px;
                background-color: #FF4B4B;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;">
                ➡️ Get Directions on Google Maps
            </a>
        </div>
        """
        directory_html_pieces.append(card_html)
        
    # Close the master scrollable container tag 
    directory_html_pieces.append("</div>")
    
    # 3. CRITICAL: Render the entire string to the browser DOM in ONE single call
    full_directory_html = "".join(directory_html_pieces)
    st.markdown(full_directory_html, unsafe_allow_html=True)
    
    # 4. Keep your existing auto-scroll animation block exactly as it was
    if st.session_state.scroll_trigger and st.session_state.selected_doctor_idx is not None:
        target_idx = st.session_state.selected_doctor_idx
        
        st.components.v1.html(
            f"""
            <script>
                const container = window.parent.document.getElementById("scrollable-directory");
                const targetCard = window.parent.document.getElementById("doc-card-{target_idx}");
                if (container && targetCard) {{
                    container.scrollTo({{
                        top: targetCard.offsetTop - container.offsetTop - 10,
                        behavior: "smooth"
                    }});
                }}
            </script>
            """,
            height=0,
            width=0
        )
        st.session_state.scroll_trigger = False