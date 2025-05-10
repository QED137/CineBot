#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# this is dummy testing file fir indexing and vector embeddings
#  after completion i will write to generator 
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


from config import settings
from langchain_community.graphs import Neo4jGraph
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from typing import Optional, List
from urllib.parse import quote
import logging



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize Config ---
NEO4J_URI = settings.NEO4J_URI
NEO4J_USERNAME = settings.NEO4J_USERNAME
NEO4J_PASSWORD = settings.NEO4J_PASSWORD
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_ENDPOINT = settings.OPENAI_ENDPOINT
TMDB_API_KEY = settings.TMDB_API_KEY
OMDB_API = settings.OMDB_API
BASE_URL = "https://api.themoviedb.org/3"
OMDB_URL = f"http://www.omdbapi.com/?apikey={OMDB_API}&"

# --- Load CLIP Model (Global) ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Neo4j Connection ---
def connect_neo() -> Neo4jGraph:
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database="neo4j"
    )
kg = connect_neo()

# --- Vector Index Management ---
def create_vector_indexes() -> None:
    
    kg.query("""
    CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
    FOR (m:Movie) ON (m.taglineEmbedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
    """)
    kg.query("""
    CREATE VECTOR INDEX movie_poster_embeddings IF NOT EXISTS
    FOR (m:Movie) ON (m.posterEmbedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 512, `vector.similarity_function`: 'cosine'}}
    """)
    logger.info("✅ Vector indexes created/verified.")

# --- Embedding Generation ---
def generate_tagline_embeddings() -> None:
    kg = connect_neo()
    kg.query("""
    MATCH (m:Movie) WHERE m.tagline IS NOT NULL AND m.taglineEmbedding IS NULL
    WITH m, genai.vector.encode(
        m.tagline, 
        "OpenAI", 
        {token: $apiKey, endpoint: $endpoint}
    ) AS embedding
    SET m.taglineEmbedding = embedding
    """, params={"apiKey": OPENAI_API_KEY, "endpoint": OPENAI_ENDPOINT})
    logger.info("✅ Tagline embeddings generated.")

# def generate_image_embedding(image_url: str) -> Optional[List[float]]:
#     try:
#         response = requests.get(image_url, stream=True, timeout=10)
#         response.raise_for_status()
#         image = Image.open(response.raw).convert("RGB")

#         inputs = clip_processor(images=image, return_tensors="pt")
#         with torch.no_grad():
#             embeddings = clip_model.get_image_features(**inputs)
#             embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

#         return embeddings[0].tolist()
#     except Exception as e:
#         logger.error(f"❌ Failed to generate image embedding: {e}")
#         return None



# --- API Integrations ---
# def get_movie_poster(title: str) -> Optional[str]:
#     try:
#         url = f"{OMDB_URL}t={quote(title)}"
#         response = requests.get(url)
#         data = response.json()
#         return data.get('Poster')
#     except Exception as e:
#         logger.error(f"OMDb API error: {e}")
#         return None

'''
the database has link for poster and trailor 
'''

# def get_trailer_key(movie_id: int) -> Optional[str]:
#     try:
#         url = f"{BASE_URL}/movie/{movie_id}/videos"
#         params = {"api_key": TMDB_API_KEY}
#         response = requests.get(url, params=params)
#         response.raise_for_status()

#         videos = response.json().get("results", [])
#         for video in videos:
#             if video.get("site") == "YouTube" and video.get("type") == "Trailer" and video.get("official"):
#                 return video.get("key")
#         return videos[0].get("key") if videos else None
#     except Exception as e:
#         logger.error(f"TMDb API error: {e}")
#         return None

# --- Update Neo4j ---
# def update_movie_with_poster(title: str) -> None:
#     """Fetch poster and update Neo4j with its embedding (and tagline if missing)."""
#     kg = connect_neo()

#     # Ensure the movie node exists
#     kg.query("""
#     MERGE (m:Movie {title: $title})
#     ON CREATE SET m.created = timestamp()
#     """, params={"title": title})
#     logger.info(f"✅ Movie '{title}' ensured in DB.")

#     # Fetch OMDB data
#     try:
#         url = f"{OMDB_URL}t={quote(title)}"
#         data = requests.get(url).json()
#     except Exception as e:
#         logger.error(f"Failed OMDB fetch: {e}")
#         return

#     tagline = data.get("Tagline") or data.get("Plot") or "No tagline available"
#     poster_url = data.get("Poster")

#     if not poster_url:
#         logger.warning(f"No poster found for {title}")
#         return

#     embedding = generate_image_embedding(poster_url)
#     if not embedding:
#         return

#     # Update the DB
#     kg.query("""
#     MATCH (m:Movie {title: $title})
#     SET m.posterUrl = $posterUrl,
#         m.posterEmbedding = $posterVec,
#         m.tagline = COALESCE(m.tagline, $tagline)
#     """, params={
#         "title": title,
#         "posterUrl": poster_url,
#         "posterVec": embedding,
#         "tagline": tagline
#     })
#     logger.info(f"✅ Poster and embedding updated for '{title}'.")

### new poster embbedding fucntion poster link is taken for the databse itself
def generate_image_embedding(kg: Neo4jGraph, tmdb_id: int) -> Optional[List[float]]:
    try:
        # Step 1: Query Neo4j for poster_url
        result = kg.query("""
            MATCH (m:Movie {tmdb_id: $tmdb_id})
            RETURN m.poster_url AS poster_url
        """, params={"tmdb_id": tmdb_id})

        if not result or not result[0]["poster_url"]:
            logger.warning(f"Poster URL not found for movie {tmdb_id}")
            return None

        image_url = result[0]["poster_url"]
        logger.info(f"📥 Downloading poster from: {image_url}")

        # Step 2: Download and embed image
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = clip_model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings[0].tolist()

    except Exception as e:
        logger.error(f"❌ Failed to generate image embedding for movie {tmdb_id}: {e}")
        return None




# --- Embedding Viewer ---

def print_movie_embeddings(title: str):
    kg = connect_neo()
    result = kg.query("""
        MATCH (m:Movie {title: $title})
        RETURN m.title AS title, m.posterEmbedding AS posterVec, m.taglineEmbedding AS taglineVec
    """, params={"title": title})

    if not result:
        logger.warning(f"⚠️ Movie '{title}' not found in Neo4j.")
        return

    movie = result[0]
    print(f"🎬 Movie: {movie['title']}")
    print(f"🖼️ Poster Embedding (first 10 dims): {movie['posterVec'][:10] if movie['posterVec'] else '❌ Not available'}")
    print(f"💬 Tagline Embedding (first 10 dims): {movie['taglineVec'][:10] if movie['taglineVec'] else '❌ Not available'}")

def testing_image_embedding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load and preprocess image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

    # Generate embedding
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    vector = image_features[0].cpu().tolist()
    print("📷 Image embedding (first 10 dims):", vector[:10])
    return vector
##the folloing function is testing the simirlyty finidng on the basis of input image vector embeddings
def searching_movie_like_input(vector):
    if not vector or not isinstance(vector, list):
        raise ValueError("❌ Embedding must be a non-empty list.")

    query = """
    CALL db.index.vector.queryNodes('movie_poster_embeddings', $topK, $embedding)
    YIELD node, score
    RETURN node.title AS title, node.poster_url AS poster, score
    ORDER BY score DESC
    """

    return kg.query(query, params={
        "embedding": vector,
        "topK": 6
    })

# the fucntion above is working
        

# --- Main Workflow ---
def main():
    # create_vector_indexes()
    # generate_tagline_embeddings()
    # update_movie_with_poster("Titanic")

    # trailer_key = get_trailer_key(27205)
    # if trailer_key:
    #     print(f"🎬 Trailer: https://www.youtube.com/watch?v={trailer_key}")

    # print_movie_embeddings("Titanic")
    vec = testing_image_embedding()
    if vec:
        print("✅ Image embedding generated.")
        results = searching_movie_like_input(vec)
        for movie in results:
            print(f"{movie['title']} (score: {movie['score']:.3f})")
    else:
        print("❌ Embedding was None.")
    
    
    # print("Trying to write to the database")
    #print("checking query connection- ", query2)
    
    #writeMovie_to_DB()
    
    # vect=testing_image_embedding()
    # print(vect)
 

if __name__ == '__main__':
    main()
