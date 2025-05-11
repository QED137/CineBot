# # config/settings.py
# import os
# from dotenv import load_dotenv
# # --- Database ---
# load_dotenv()
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# # --- APIs ---
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TMDB_API_KEY = os.getenv("TMDB_API_KEY")
# OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# # Embedding Model Configuration
# TEXT_EMBEDDING_MODEL_NAME = os.getenv("TEXT_EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# VISION_EMBEDDING_MODEL_NAME = os.getenv("VISION_EMBEDDING_MODEL_NAME", "openai/clip-vit-base-patch32")

# if __name__ == "main":
#      # --- Basic Validation  ---
#      if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, TMDB_API_KEY]):
#           print("Warning: Essential Neo4j or TMDB secrets seem to be missing!")
#      # raise an error here if they are absolutely required
#      # raise ValueError("Missing critical configuration secrets!")
# print(NEO4J_USERNAME)   

# config/settings.py
import os
#from dotenv import load_dotenv # You can keep this for local development

# This line loads variables from a .env file *if it exists*.
# In Codespaces (where you haven't created a .env file), it likely does nothing.
# Crucially, it usually *doesn't* overwrite variables already set in the environment.
#load_dotenv()

# --- This is the key part ---
# os.getenv() reads directly from the environment variables.
# Since GitHub Codespace Secrets *are* environment variables, these lines will pick them up automatically.

import os

try:
    import streamlit as st
    USE_STREAMLIT_SECRETS = True
except ImportError:
    USE_STREAMLIT_SECRETS = False

def get_secret(key: str) -> str:
    if USE_STREAMLIT_SECRETS and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

NEO4J_URI = get_secret("NEO4J_URI")
NEO4J_USERNAME = get_secret("NEO4J_USERNAME")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_ENDPOINT = get_secret("OPENAI_ENDPOINT") or "https://api.openai.com/v1"
TMDB_API_KEY = get_secret("TMDB_API_KEY")
OMDB_API = get_secret("OMDB_API")



# --- End key part ---

# Your validation logic can remain the same
if not NEO4J_URI or not NEO4J_PASSWORD:
    print("Warning: Neo4j URI or Password not found in environment.")
# ... etc ...
def main():
    print("Configuration loaded successfully.")
    from config import settings
    print("✅ OMDB_API loaded:", settings.OMDB_API)
    
    print("✅ OMDB_API =", os.getenv("OMDB_API"))
    print("✅ All ENV keys:", list(os.environ.keys()))


if __name__ == "__main__":
    main()
    