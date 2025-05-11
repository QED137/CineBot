# this files generate .streamlit/secretes.toml for cloud deployment because cloud deployment needs only 
# read pssowrds and key from secretes.toml


import os

def write_streamlit_secrets_from_env():
    os.makedirs(".streamlit", exist_ok=True)
    with open(".streamlit/secrets.toml", "w") as f:
        f.write(f"""
OPENAI_API_KEY = "{os.environ.get('OPENAI_API_KEY', '')}"
OPENAI_ENDPOINT = "{os.environ.get('OPENAI_ENDPOINT', '')}"
NEO4J_URI = "{os.environ.get('NEO4J_URI', '')}"
NEO4J_USERNAME = "{os.environ.get('NEO4J_USERNAME', '')}"
NEO4J_PASSWORD = "{os.environ.get('NEO4J_PASSWORD', '')}"
TMDB_API_KEY = "{os.environ.get('TMDB_API_KEY', '')}"
OMDB_API = "{os.environ.get('OMDB_API', '')}"
""")
