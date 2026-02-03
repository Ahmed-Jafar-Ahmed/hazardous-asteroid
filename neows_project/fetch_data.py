"""
Usage: to fetch data from a URL and save it to a specified file path.
Loading data to data/raw/ directory.
"""
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import time

load_dotenv()

""" Configuration """
API_URL = "https://api.nasa.gov/neo/rest/v1/neo/browse"
API_KEY = os.getenv("API_KEY")
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

all_data = []
page = 0
while True:
    params = {
        "api_key": API_KEY,
        "page": page,
        "size": 200  # Max size per API documentation
    }
    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    data = response.json()

    neo = data.get("near_earth_objects", [])
    if not neo:
        break
    
    # Flatten nested JSON structure
    df = pd.json_normalize(neo, sep="_")
    all_data.append(df)

    print(f"Fetched page {page} with {len(df)} records.")
    page += 1
    if page >= data["page"]["total_pages"]:
        break

    time.sleep(0.2)  # To respect API rate limits

# Combine all dataframes
final_df = pd.concat(all_data, ignore_index=True)
file_path = DATA_DIR / "asteroids_data.csv"

# Save to CSV
final_df.to_csv(file_path, index=False)

print(f"Data saved to {file_path}")