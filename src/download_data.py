import pandas as pd
import requests
import os
import time

BASE_URL = "https://health.data.ny.gov/resource/82xm-y6g8.csv"
OUTPUT_DIR = "data/hospitals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_top_10_hospitals():
    """
    Pulls facility names and counts from the API to find the 10 largest hospitals.
    Uses SoQL aggregation - like SQL GROUP BY.
    """
    print("Finding top 10 hospitals by record count...")

    params = {
        "$select": "facility_name, count(*) as record_count",
        "$group": "facility_name",
        "$order": "record_count DESC",
        "$limit": 10
    }

    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code} - {response.text[:200]}")
        return []

    df = pd.read_csv(pd.io.common.StringIO(response.text))
    print(df.to_string())
    return df["facility_name"].tolist()


def download_hospital(hospital_name, client_number):
    print(f"\nDownloading Client {client_number}: {hospital_name}...")

    params = {
        "$where": f"facility_name='{hospital_name}'",
        "$limit": 50000
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f"  ERROR: {response.status_code} - {response.text[:200]}")
        return

    filepath = os.path.join(OUTPUT_DIR, f"client_{client_number}.csv")
    with open(filepath, "wb") as f:
        f.write(response.content)

    df = pd.read_csv(filepath)
    print(f"  Saved {len(df)} records to {filepath}")
    time.sleep(2)


if __name__ == "__main__":
    hospitals = get_top_10_hospitals()

    if not hospitals:
        print("Could not retrieve hospital list. Exiting.")
    else:
        for i, hospital in enumerate(hospitals, start=1):
            download_hospital(hospital, i)

        print("\nDone. Check data/hospitals/ for your files.")