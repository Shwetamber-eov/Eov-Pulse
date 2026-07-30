import json
import requests
import heapq
import math
from geopy.distance import geodesic
import os

# Replace with your actual free SerpApi Key
SERP_API_KEY =os.getenv("SERP_API_KEY")

def fetch_local_doctors(specialty, location, user_coordinates):
    print(f"Querying SerpApi REST Endpoint for: '{specialty} clinic near {location}'...")
    
    # SerpApi's direct web routing parameters
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_maps",
        "q": f"{specialty} clinic in {location}",
        "hl": "en",
        "api_key": SERP_API_KEY
    }

    try:
        # Fire standard HTTPS web call directly to the engine
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"❌ SerpApi server rejected query. Code: {response.status_code}")
            print(f"Server message: {response.text}")
            return []
        results = response.json()
        
        # Check if local map queries are returned in the payload array
        if "local_results" not in results:
            print("⚠️ No local listings found or API allocation credit exhausted.")
            return []
            
        raw_listings = results["local_results"]
        # print(json.dumps(raw_listings[0], indent=2))

        no_of_doctors=5
        distance_threshold=5    #in km
        #filtering list based on distance threshold
        valid_listings = [item for item in raw_listings if round(geodesic(user_coordinates,(item.get("gps_coordinates", {}).get("latitude"), item.get("gps_coordinates", {}).get("longitude"))).km,2) <= distance_threshold]
        #sorting list based on ratings and reviews count
        top_n_listings = heapq.nlargest(no_of_doctors, valid_listings, key=lambda x: x.get("rating", 0) * math.log10(x.get("reviews", 0) + 1))
        ai_ready_doctors = []
        # Restrict parameters to top 5 hits to save context token fees in your LLM pipeline
        for item in top_n_listings:
                cleaned_profile = {
                    "name": item.get("title"),
                    "category": item.get("type") or specialty,
                    "phone": item.get("phone", "N/A"),
                    "address": item.get("address"),
                    "latitude": item.get("gps_coordinates", {}).get("latitude"),
                    "longitude": item.get("gps_coordinates", {}).get("longitude"),
                    "rating": f"{item.get('rating', 'N/A')} stars",
                    "reviews_count": item.get("reviews", 0),
                    "website": item.get("website", "None"),
                    "open_state": item.get("operating_hours", {}).get("open_now", "Unknown")}
                ai_ready_doctors.append(cleaned_profile)
        return ai_ready_doctors

    except Exception as e:
        print(f"Workflow execution pipeline failed: {e}")
        return []

# Run validation lookup trace
# doctors_json_payload = fetch_local_doctors(specialty="Dermatologist", location="Erandwane, Pune")

import json

def extract_specialists(llm_response):
    if llm_response is None:
        return []
    if isinstance(llm_response, str):
        llm_response = json.loads(llm_response)

    # If a single dict is passed, wrap it in a list
    if isinstance(llm_response, dict):
        llm_response = [llm_response]

    specialists = [
        item.get("specialist","")
        for item in llm_response
    ]

    specialists = [
        s for s in specialists
        if s and str(s).strip().lower() not in {
            "", "none", "n/a", "no action required"
        }
    ]

    return specialists