# config/settings.py
import os
from dotenv import load_dotenv
# --- Database ---
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- APIs ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

if __name__ == "main":
     # --- Basic Validation (Optional but good practice) ---
     if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, TMDB_API_KEY]):
          print("Warning: Essential Neo4j or TMDB secrets seem to be missing!")
     # You could raise an error here if they are absolutely required
     # raise ValueError("Missing critical configuration secrets!")
     print(NEO4J_USERNAME)   

