import streamlit as st
import folium
from streamlit_folium import st_folium
import urllib.parse


def gmap(ai_ready_doctors):
    if not ai_ready_doctors:
        st.warning("No nearby specialists found within the search radius.")
        return

    start_lat = ai_ready_doctors[0]["latitude"]
    start_lon = ai_ready_doctors[0]["longitude"]

    st.subheader("🗺️ Interactive Doctor Map")

    col1, col2 = st.columns([2, 1])

    if "selected_doctor_idx" not in st.session_state:
        st.session_state.selected_doctor_idx = None
    if "scroll_trigger" not in st.session_state:
        st.session_state.scroll_trigger = False

    with col1:
        st.caption("Click a marker to see details on the right")
        m = folium.Map(location=[start_lat, start_lon], zoom_start=13)

        for idx, doc in enumerate(ai_ready_doctors):
            icon_html = """
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
                options={"id": idx}
            ).add_to(m)

        # KEY FIX: only report back the piece of state we actually use.
        # Without returned_objects, st_folium reports zoom/bounds/center on
        # every pan or zoom, and any change in a component's return value
        # forces Streamlit to rerun the whole script.
        map_data = st_folium(
            m,
            width=700,
            height=500,
            key="doctor_map",
            returned_objects=["last_object_clicked"]
        )

        if map_data and map_data.get("last_object_clicked"):
            clicked_lat = map_data["last_object_clicked"]["lat"]
            clicked_lng = map_data["last_object_clicked"]["lng"]
            for idx, doc in enumerate(ai_ready_doctors):
                if abs(doc["latitude"] - clicked_lat) < 1e-6 and abs(doc["longitude"] - clicked_lng) < 1e-6:
                    if st.session_state.selected_doctor_idx != idx:
                        st.session_state.selected_doctor_idx = idx
                        st.session_state.scroll_trigger = True
                        st.rerun()
                    break

    with col2:
        st.subheader("Doctor Directory")
        selected_idx = st.session_state.selected_doctor_idx
        cards_html = ""
        for idx, doc in enumerate(ai_ready_doctors):
            is_selected = idx == selected_idx
            border = "2px solid #FF4B4B" if is_selected else "1px solid #e6e6e6"
            bg = "#FFF5F5" if is_selected else "white"
            query_string = urllib.parse.quote(f"{doc['name']}, {doc['address']}")
            gmap_link = f"https://google.com/maps/search/?api=1&query={query_string}"
            open_state = doc.get("open_state", "Unknown")
            open_badge = (
                "🟢 Open now" if open_state is True
                else "🔴 Closed" if open_state is False
                else "Hours unknown"
            )

            cards_html += f"""
            <div
                id="doc-card-{idx}"
                style="
                    border:{border};
                    background:{bg};
                    padding:15px;
                    margin:10px 0;
                    border-radius:8px;
                    font-family:sans-serif;
                "
            >
                <h3 style="margin:0 0 6px 0;">{doc['name']}</h3>
                <p style="margin:2px 0;">⭐ {doc['rating']} &nbsp;·&nbsp; {doc.get('reviews_count', 0)} reviews</p>
                <p style="margin:2px 0;">📍 {doc['address']}</p>
                <p style="margin:2px 0;">☎️ {doc.get('phone', 'N/A')} &nbsp;·&nbsp; {open_badge}</p>
                <a
                    href="{gmap_link}"
                    target="_blank"
                    style="
                        display:inline-block;
                        margin-top:8px;
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

        scroll_script = ""
        if selected_idx is not None:
            scroll_script = f"""
            <script>
            const target = document.getElementById("doc-card-{selected_idx}");
            if (target) {{
                target.scrollIntoView({{ behavior: "smooth", block: "center" }});
            }}
            </script>
            """

        st.components.v1.html(
            f"""
            <div
                id="scrollable-directory"
                style="height:500px; overflow-y:auto; padding-right:10px;"
            >{cards_html}</div>
            {scroll_script}
            """,
            height=520,
            scrolling=False
        )