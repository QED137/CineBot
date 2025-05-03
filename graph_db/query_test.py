from config import settings
from langchain_community.graphs import Neo4jGraph
import requests
# Load environment settings
NEO4J_URI = settings.NEO4J_URI
NEO4J_USERNAME = settings.NEO4J_USERNAME
NEO4J_PASSWORD = settings.NEO4J_PASSWORD
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_ENDPOINT= settings.OPENAI_ENDPOINT
OMDB_API=settings.OMDB_API
OMDB_URL = f"http://www.omdbapi.com/?apikey={OMDB_API}&"

# Initialize Neo4j LangChain connection
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j"
)

# ✅ FIXED Cypher syntax for creating vector index
def create_tagline_embeddings():
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



def fetch_movie_from_omdb(title):
    url = OMDB_URL + f"t={title}"
    response = requests.get(url)
    data = response.json()
    return data

# Example test
movie_data = fetch_movie_from_omdb("Inception")
print("🎬 Title:", movie_data.get("Title"))
print("📅 Year:", movie_data.get("Year"))
print("🧾 Plot:", movie_data.get("Plot"))
print("🖼️ Poster:", movie_data.get("Poster"))
