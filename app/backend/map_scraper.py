import json
import requests
import heapq
import math
import re
from geopy.distance import geodesic

# Replace with your actual free SerpApi Key
SERP_API_KEY = "API_KEY"

def fetch_local_doctors(specialty, location, user_coordinates):
    print(f"Querying SerpApi REST Endpoint for: '{specialty} clinic in {location}'...")
    
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

def extract_specialists(text):
    # Regex looks for "Specialist Referral:" and captures everything until the end of the line
    match = re.search(r"Specialist Referral:\s*(.*)", text, re.IGNORECASE)
    
    if match:
        specialist_string = match.group(1).strip()
        
        # Handle cases where no specialist is needed
        if specialist_string.lower() in ["none", "n/a", "no action required"]:
            return []
        
        # Split by commas or slashes if multiple specialists were listed
        specialists = [s.strip() for s in re.split(r'[,/]', specialist_string) if s.strip()]
        return specialists
    
    return []

# Usage
# specialist_list = extract_specialists(llm_response)

# print("\n🚀 Clean structured output for your AI Workflow:")
# print(json.dumps(doctors_json_payload, indent=2))
