# from sentence_transformers import SentenceTransformer
# from config import settings
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from config import settings
# import numpy as np
# import logging
# from typing import List, Optional
# import requests # To fetch image for embedding during load
# from PIL import Image
# import io

# # Keep existing text embedding functions
# from multimodal.vision_model import get_clip_image_embedding # Import image embedder
# from multimodal.image_processor import load_image_from_bytes

# import logging
# from typing import List

# log = logging.getLogger(__name__)

# _model = None
# _text_model = None
# # Poster base URL for TMDb
# POSTER_BASE_URL_LOAD = "https://image.tmdb.org/t/p/w185" # Use a consistent size

# def get_embedding_model() -> SentenceTransformer:
#     """Loads and returns the SentenceTransformer model."""
#     global _model
#     if _model is None:
#         log.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
#         try:
#             # Instantiate the SentenceTransformer model
#             _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
#             log.info("Embedding model loaded successfully.")
#         except Exception as e:
#             log.error(f"Failed to load embedding model '{settings.EMBEDDING_MODEL_NAME}': {e}")
#             raise RuntimeError(f"Could not load embedding model: {e}") from e
#     return _model

# def generate_embedding(text: str) -> List[float]:
#     """Generates a vector embedding for the given text."""
#     if not text or not isinstance(text, str):
#         log.warning("Attempted to generate embedding for empty or non-string input.")
#         # Return a zero vector of the expected dimension or handle as error
#         # Getting dimension requires loading the model first.
#         try:
#             model = get_embedding_model()
#             dimension = model.get_sentence_embedding_dimension()
#             return [0.0] * dimension
#         except Exception:
#              return [] # Fallback if model can't load

#     try:
#         model = get_embedding_model()
#         embedding = model.encode(text, convert_to_numpy=True)
#         return embedding.tolist() # Convert numpy array to list for JSON/Neo4j compatibility
#     except Exception as e:
#         log.error(f"Error generating embedding: {e}")
#         # Depending on requirements, return empty list, zero vector, or re-raise
#         return []

# def get_text_embedding_model() -> SentenceTransformer:
#     """Loads and returns the SentenceTransformer model."""
#     global _text_model
#     if _text_model is None:
#         log.info(f"Loading text embedding model: {settings.TEXT_EMBEDDING_MODEL_NAME}")
#         try:
#             _text_model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL_NAME)
#             log.info("Text embedding model loaded successfully.")
#         except Exception as e:
#             log.error(f"Failed to load text embedding model '{settings.TEXT_EMBEDDING_MODEL_NAME}': {e}")
#             raise RuntimeError(f"Could not load text embedding model: {e}") from e
#     return _text_model

# def generate_text_embedding(text: str) -> List[float]:
#     """Generates a vector embedding for the given text."""
#     if not text or not isinstance(text, str):
#         log.warning("Attempted to generate text embedding for empty or non-string input.")
#         try:
#             model = get_text_embedding_model()
#             dimension = model.get_sentence_embedding_dimension()
#             return [0.0] * dimension
#         except Exception:
#              return []

#     try:
#         model = get_text_embedding_model()
#         embedding = model.encode(text, convert_to_numpy=True)
#         return embedding.tolist()
#     except Exception as e:
#         log.error(f"Error generating text embedding: {e}")
#         return []

# # --- NEW: Function to generate image embedding during data load ---
# def generate_image_embedding_from_url(poster_path: Optional[str]) -> List[float]:
#     """
#     Fetches poster image from TMDb url (using poster_path) and generates embedding.
#     Used during the data loading script.
#     """
#     if not poster_path:
#         log.debug("No poster path provided, skipping image embedding generation.")
#         return []

#     image_url = f"{POSTER_BASE_URL_LOAD}{poster_path}"
#     log.debug(f"Fetching image for embedding from: {image_url}")

#     try:
#         response = requests.get(image_url, timeout=10) # Add timeout
#         response.raise_for_status() # Check for HTTP errors

#         image_bytes = response.content
#         image = load_image_from_bytes(image_bytes)

#         # Generate embedding using the vision model function
#         embedding = get_clip_image_embedding(image)
#         if embedding:
#              log.debug(f"Successfully generated embedding for poster: {poster_path}")
#         else:
#              log.warning(f"Failed to generate embedding for poster: {poster_path}")
#         return embedding

#     except requests.exceptions.RequestException as e:
#         log.warning(f"Failed to fetch image from {image_url}: {e}")
#         return []
#     except ValueError as e: # Catch image loading errors
#         log.warning(f"Failed to load image from {image_url}: {e}")
#         return []
#     except Exception as e:
#         log.error(f"Unexpected error generating image embedding for {poster_path}: {e}")
#         return []
# # Example usage
# if __name__ == "__main__":
#     plot = "An orphaned boy enrolls in a school of wizardry, where he learns the truth about himself, his family and the terrible evil that haunts the magical world."
#     embedding = generate_embedding(plot)
#     if embedding:
#         print(f"Generated embedding of dimension: {len(embedding)}")
#         # print(embedding[:10]) # Print first 10 dimensions
#     else:
#         print("Failed to generate embedding.")

#     empty_emb = generate_embedding("")
#     print(f"Empty embedding dimension: {len(empty_emb)}")

##################################################################################################################### # 
# this file contains text embedder for tagline and image embbedder for posters to make sure the multimodal search                                                                                                                    #
#                                                                                                                     #
#                                                                                                                     #
######################################################################################################################



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


### new poster embbedding fucntion poster link is taken for the databse itself
import os
import torch
import requests
from PIL import Image
from typing import List, Dict, Optional
from langchain_community.graphs import Neo4jGraph
from transformers import CLIPProcessor, CLIPModel
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# Replace with your actual settings
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

def get_poster_urls_batch(kg: Neo4jGraph, skip: int, limit: int = 100) -> List[Dict]:
    return kg.query("""
        MATCH (m:Movie)
        WHERE m.poster_url IS NOT NULL AND m.poster_embedding IS NULL
        RETURN m.tmdb_id AS id, m.poster_url AS url
        SKIP $skip
        LIMIT $limit
    """, params={"skip": skip, "limit": limit})

def generate_batch_image_embeddings(movie_posters: List[Dict]) -> Dict[int, Optional[List[float]]]:
    embeddings = {}
    for poster in movie_posters:
        movie_id = poster["id"]
        image_url = poster["url"]
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                embeddings[movie_id] = features[0].cpu().tolist()

        except Exception as e:
            logger.warning(f"⚠️ Movie {movie_id} failed: {e}")
            embeddings[movie_id] = None

        torch.cuda.empty_cache()
    return embeddings

def store_embeddings_in_neo4j(kg: Neo4jGraph, embeddings: Dict[int, List[float]]) -> None:
    for movie_id, vector in embeddings.items():
        if vector:
            kg.query("""
                MATCH (m:Movie {tmdb_id: $tmdb_id})
                SET m.poster_embedding = $embedding
            """, params={
                "tmdb_id": movie_id,
                "embedding": vector
            })

def embed_all_posters_in_chunks(kg: Neo4jGraph, batch_size: int = 100, max_movies: int = 10000):
    resume_file = "last_successful_skip.txt"
    start = 0
    if os.path.exists(resume_file):
        with open(resume_file) as f:
            start = int(f.read())

    for skip in range(start, max_movies, batch_size):
        logger.info(f"🔁 Processing batch {skip} to {skip + batch_size - 1}")
        movies = get_poster_urls_batch(kg, skip=skip, limit=batch_size)
        if not movies:
            logger.info("✅ No more movies to process.")
            break

        embeddings = generate_batch_image_embeddings(movies)
        store_embeddings_in_neo4j(kg, embeddings)

        with open(resume_file, "w") as f:
            f.write(str(skip + batch_size))

        logger.info(f"✅ Completed batch {skip}–{skip + batch_size - 1}")

# Uncomment to run the embedding process
# embed_all_posters_in_chunks(kg, batch_size=100, max_movies=10000)



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

# --- Main Workflow ---
def main():
    # create_vector_indexes()
    # generate_tagline_embeddings()
    # update_movie_with_poster("Titanic")

    # trailer_key = get_trailer_key(27205)
    # if trailer_key:
    #     print(f"🎬 Trailer: https://www.youtube.com/watch?v={trailer_key}")

    # print_movie_embeddings("Titanic")
    print("Trying to write to the database")
    
    result=kg.query(
    """
    MATCH(n) 
    RETURN COUNT(n)
    """
    )
    print("checking query connection- ", result)
    print("try to write into the database")
    #writeMovie_to_DB()
    


if __name__ == '__main__':
    main()



