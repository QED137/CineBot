# test_api.py (IMPROVED WITH EXPLICIT CONTENT-TYPE)

import os
import requests
import mimetypes # <-- Add this import
from dotenv import load_dotenv

print("--- TESTING IMAGE API WITH EXPLICIT CONTENT-TYPE HEADER ---")

load_dotenv()
api_token = os.getenv("HF_TOKEN")

# We will use the -large model as it's a good, responsive choice.
api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-large-patch14"
image_path = "/home/graviton/Downloads/titanic.jpg" 

if not api_token:
    print("\n--- ERROR: Could not find HF_API_TOKEN. ---")
    exit()
else:
    print(f"Successfully loaded token starting with: {api_token[:7]}...")

if not os.path.exists(image_path):
    print(f"\n--- ERROR: Cannot find the image file at: {image_path} ---")
    exit()
else:
     print(f"Found image file at: {image_path}")

try:
    # --- THIS IS THE CRITICAL CHANGE ---
    # 1. Guess the image's MIME type (e.g., 'image/jpeg') from its filename
    content_type, _ = mimetypes.guess_type(image_path)
    if content_type is None:
        # Fallback if the type can't be guessed
        content_type = "application/octet-stream"
    print(f"Detected Content-Type: {content_type}")

    # 2. Add the detected Content-Type to the headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": content_type 
    }
    
    with open(image_path, "rb") as f:
        image_data = f.read()

    print("\nSending request with explicit Content-Type header...")
    response = requests.post(api_url, headers=headers, data=image_data, timeout=30)
    
    print(f"--> Received HTTP Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("\n--- SUCCESS! ---")
        embedding = response.json()[0][:5]
        print(f"Received embedding successfully. First 5 values: {embedding}")
    else:
        print("\n--- FAILURE ---")
        print("API call failed. Server response:")
        print(response.text)

except Exception as e:
    print(f"\n--- SCRIPT ERROR ---")
    print(f"An error occurred while running the script: {e}")