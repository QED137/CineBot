# this files generate .streamlit/secretes.toml for cloud deployment because cloud deployment needs only 
# read pssowrds and key from secretes.toml


import os

def write_streamlit_secrets_from_env():
    os.makedirs(".streamlit", exist_ok=True)  # ✅ Ensure directory exists

    with open(".streamlit/secrets.toml", "w") as f:
        f.write(f"""
NEO4J_URI = "{os.getenv("NEO4J_URI", "")}"
NEO4J_USERNAME = "{os.getenv("NEO4J_USERNAME", "")}"
NEO4J_PASSWORD = "{os.getenv("NEO4J_PASSWORD", "")}"
NEO4J_DATABASE = "{os.getenv("NEO4J_DATABASE", "neo4j")}"

OPENAI_API_KEY = "{os.getenv("OPENAI_API_KEY", "")}"
OPENAI_ENDPOINT = "{os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")}"

TMDB_API_KEY = "{os.getenv("TMDB_API_KEY", "")}"
OMDB_API = "{os.getenv("OMDB_API", "")}"
""")
