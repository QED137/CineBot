import os
# --- Database ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- APIs ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Kaggle (if needed by scripts) ---
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

REQUIRED_SECRETS = [
    "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
    "OPENAI_API_KEY", "TMDB_API_KEY", "OMDB_API_KEY"
]
missing_secrets = [secret for secret in REQUIRED_SECRETS if not globals().get(secret)]
if missing_secrets:
    # Log this or raise an error depending on how critical it is
    print(f"Warning: Missing required environment variables/secrets: {', '.join(missing_secrets)}")

print('NEO4J_URL')

