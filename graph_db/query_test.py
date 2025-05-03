from config import settings
from langchain_community.graphs import Neo4jGraph
import requests
from urllib.parse import quote
# Load environment settings
NEO4J_URI = settings.NEO4J_URI
NEO4J_USERNAME = settings.NEO4J_USERNAME
NEO4J_PASSWORD = settings.NEO4J_PASSWORD
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_ENDPOINT= settings.OPENAI_ENDPOINT
OMDB_API=settings.OMDB_API
OMDB_URL = f"http://www.omdbapi.com/?apikey={OMDB_API}&"

# Initialize Neo4j LangChain connection
def connect_neo():
    kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j"
    )
    return kg

# ✅ FIXED Cypher syntax for creating vector index
def create_tagline_embeddings():
    kg = connect_neo()
    kg.query(
    """
    CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
    FOR (m:Movie) ON (m.taglineEmbedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }
    }
    """)
def generate_embedding_tagline():
    kg = connect_neo()
    OPENAI_API_KEY = settings.OPENAI_API_KEY
    OPENAI_ENDPOINT= settings.OPENAI_ENDPOINT
    kg.query("""
    MATCH (movie:Movie) WHERE movie.tagline IS NOT NULL
    WITH movie, genai.vector.encode(
        movie.tagline, 
        "OpenAI", 
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS vector
    CALL db.create.setNodeVectorProperty(movie, "taglineEmbedding", vector)
    """, 
    params={"openAiApiKey":OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT} )

# ✅ List all vector indexes
#result = kg.query("SHOW VECTOR INDEXES")
#print(result)

#generate_embedding_tagline()

#test tagline embeedings 
#kg=connect_neo()
# result = kg.query(
#     """
#     MATCH (m:Movie) 
#     WHERE m.tagline IS NOT NULL
#     RETURN m.tagline, m.taglineEmbedding
#     LIMIT 1 
#     """
# )

# print(result[0]['m.tagline'])
# print(result[0]['m.taglineEmbedding'][:10])


def fetch_movie_from_omdb(title: str):
    encoded_title = quote(title)
    url = OMDB_URL + f"t={encoded_title}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("Response") == "True":
            return {
                "title": data.get("Title"),
                "year": data.get("Year"),
                "plot": data.get("Plot"),
                "poster": data.get("Poster"),
            }
        else:
            print(f"❌ OMDb Error: {data.get('Error')}")
            return None
    except Exception as e:
        print(f"🚨 Request failed: {e}")
        return None

# 🧪 Example usage
movie = fetch_movie_from_omdb("Inception")
if movie:
    print("🎬 Title:", movie["title"])
    print("📅 Year:", movie["year"])
    print("🧾 Plot:", movie["plot"])
    print("🖼️ Poster:", movie["poster"])
else:
    print("Movie not found or error occurred.")
