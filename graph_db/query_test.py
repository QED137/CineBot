from config import settings
from langchain_community.graphs import Neo4jGraph
import requests
from urllib.parse import quote
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
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
    
#imporitng movie data aand poster from omdbi

def moviePoster(title):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API}&t={title}"
    response=requests.get(url)
    data = response.json()
    return data.get('Poster')

# Load model once (reuse for multiple calls)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

###generating poster embedding  so that it can store vector information
def generate_image_embedding_from_url(image_url: str) -> list:
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = clip_model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)  # normalize

        return embeddings[0].tolist()  # Return as list of floats (length 512)
    except Exception as e:
        print(f"❌ Failed to generate image embedding: {e}")
        return []



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



# import requests
# from config import settings

# OMDB_API_KEY = settings.OMDB_API
# title = "Inception"

# url = f"http://www.omdbapi.com/?apikey={OMDB_API}&t={title}"
# response = requests.get(url)
# data = response.json()

# print("🔍 Raw OMDB response:", data)  # <-- Add this line

# print("🎬 Title:", data.get("Title"))
# print("📅 Year:", data.get("Year"))
# print("🧾 Plot:", data.get("Plot"))
# print("🖼️ Poster:", data.get("Poster"))
# print("✅ OMDB_API loaded:", settings.OMDB_API)



#Inception = moviePoster("Inception")
#url = f"http://www.omdbapi.com/?apikey={OMDB_API}&t={Titanic}"
#list1= generate_image_embedding_from_url(url)

poster= moviePoster("Titanic")

print(poster)
list1= generate_image_embedding_from_url(poster)
print(list1[:10])
