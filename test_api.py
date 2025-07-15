"""Test script for the headline sentiment API."""

import requests
import json

BASE_URL = "http://localhost:8016"

# Check status
response = requests.get(f"{BASE_URL}/status")
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

# Score headlines
test_headlines = [
    "Scientists Discover Cure for Major Disease",
    "Stock Market Crashes Amid Economic Fears", 
    "Local Library Hosts Book Reading Event",
    "New Technology Promises Better Future",
    "Unemployment Rates Hit Record High"
]

data = {"headlines": test_headlines}

response = requests.post(
    f"{BASE_URL}/score_headlines",
    json=data,
    headers={"Content-Type": "application/json"}
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

# Match headlines with labels
if response.status_code == 200:
    labels = response.json()["labels"]
    print("\nResults:")
    for headline, label in zip(test_headlines, labels):
        print(f"  {label}: {headline}")