import streamlit as st
import folium
from streamlit_folium import st_folium
import urllib.parse
# import pandas as pd
# Sample data matching your ai_ready_doctors structure (with lat/lon added)
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
def gmap(ai_ready_doctors):
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
            # folium.CircleMarker(
            #     location=[doc["latitude"], doc["longitude"]],
            #     radius=10,  # Slightly bigger for easier mobile clicking
            #     tooltip=doc["name"],
            #     color="#FF4B4B",
            #     fill=True,
            #     fill_color="#FF4B4B",
            #     fill_opacity=1,
            #     # CRITICAL: Save the index into the marker options dictionary
            #     options={"id": idx}  
            # ).add_to(m)
            icon_html = f"""
            <div style="
            background-color: #FF4B4B; 
            width: 14px; 
            height: 14px; 
            border-radius: 50%; 
            border: 2px solid white; 
            box-shadow: 0 0 4px rgba(0,0,0,0.4);
            cursor: pointer;">
            </div>
            """
            custom_icon = folium.DivIcon(
            html=icon_html,
            icon_size=(14, 14),
            icon_anchor=(7, 7)
            )

            folium.Marker(
            location=[doc["latitude"], doc["longitude"]],
            tooltip=doc["name"],
            icon=custom_icon,
            # Assigning an explicit integer ID inside options makes it readable by st_folium
            options={"id": idx}  
            ).add_to(m)

        # 2. Capture the map interaction payload
        map_data = st_folium(m, width=700, height=500, key="doctor_map")
    
        # 3. Detect if a marker was clicked and update the active index
        if map_data and map_data.get("last_object_clicked"):
            clicked_lat = map_data["last_object_clicked"]["lat"]
            clicked_lng = map_data["last_object_clicked"]["lng"]
            # clicked_object = map_data["last_object_clicked"]

            # if "id" in clicked_object:
            #     clicked_idx = int(clicked_object["id"])
            #     if st.session_state.selected_doctor_idx != clicked_idx:
            #         st.session_state.selected_doctor_idx = clicked_idx
            #         st.session_state.scroll_trigger = True
            #         st.rerun()
        
            # Match back coordinates to find the correct doctor index
            for idx, doc in enumerate(ai_ready_doctors):
                if abs(doc["latitude"] - clicked_lat) < 1e-6 and abs(doc["longitude"] - clicked_lng) < 1e-6:
                    if st.session_state.selected_doctor_idx != idx:
                        st.session_state.selected_doctor_idx = idx
                        st.session_state.scroll_trigger = True  # Signal col2 to scroll
                        st.rerun()

    with col2:
        st.subheader("Doctor Directory")
        selected_idx = st.session_state.selected_doctor_idx
        cards_html = ""
        for idx, doc in enumerate(ai_ready_doctors):
            is_selected = idx == selected_idx
            border = "2px solid #FF4B4B" if is_selected else "1px solid #e6e6e6"
            bg = "#FFF5F5" if is_selected else "white"
            query_string = urllib.parse.quote(f"{doc['name']}, {doc['address']}")
            gmap_link = (f"https://google.com/maps/search/?api=1&query={query_string}")
            # gmap_link = (f"https://google.com/maps/dir/?api=1&query={query_string}")

            cards_html += f"""
            <div
                id="doc-card-{idx}"
                style="
                    border:{border};
                    background:{bg};
                    padding:15px;
                    margin:10px 0;
                    border-radius:6px;
                "
            >
                <h3>{doc['name']}</h3>
                <p>⭐ {doc['rating']}</p>
                <p>📍 {doc['address']}</p>

                <a
                    href="{gmap_link}"
                    target="_blank"
                    style="
                        display:inline-block;
                        padding:8px 16px;
                        background:#FF4B4B;
                        color:white;
                        text-decoration:none;
                        border-radius:4px;
                    "
                >
                    Get Directions
                </a>
            </div>
            """
        st.components.v1.html(
            f"""
            <div
            id="scrollable-directory"
            style="
                height:500px;
                overflow-y:auto;
                padding-right:10px;
                ">{cards_html}</div>

            <script>
            const target =
                document.getElementById("doc-card-{selected_idx}");

            if(target){{
                target.scrollIntoView({{
                    behavior: "smooth",
                    block: "center"
                }});
            }}
            </script>
            """,
            height=520,
            scrolling=False
        )

# gmap(doctors)