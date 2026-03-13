#!/usr/bin/env python3
"""Quick test to check the latest error"""
import requests
import json

try:
    response = requests.post(
        "http://127.0.0.1:8000/api/chat",
        json={"message": "who directed titanic"},
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
except Exception as e:
    print(f"Error: {e}")
