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
from urllib.parse import quote
import os
from typing import List, Dict, Optional
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
    # Assuming kg is your global Neo4jGraph instance
    # Assuming settings.OPENAI_API_KEY and settings.OPENAI_ENDPOINT are correctly loaded

    logger.info("Attempting to generate tagline embeddings...")
    try:
        # First, count candidates to understand scope
        candidate_query = """
        MATCH (m:Movie)
        WHERE m.tagline IS NOT NULL AND m.tagline <> "" AND m.taglineEmbedding IS NULL
        RETURN count(m) as candidate_count
        """
        candidate_result = kg.query(candidate_query)
        candidate_count = candidate_result[0]['candidate_count'] if candidate_result and candidate_result[0] else 0
        logger.info(f"Found {candidate_count} movies with non-empty taglines needing embedding.")

        if candidate_count == 0:
            return

        # Main embedding query
        embedding_query = """
        MATCH (m:Movie)
        WHERE m.tagline IS NOT NULL
          AND m.tagline <> ""  // <--- CRITICAL: Filter out empty strings
          AND m.taglineEmbedding IS NULL
        WITH m, genai.vector.encode(
            m.tagline,
            "OpenAI",
            {
                token: $apiKey,
                endpoint: $endpoint
            }
        ) AS emb // Renamed to avoid confusion if 'embedding' is a property name
        WHERE emb IS NOT NULL AND size(emb) > 0 // Ensure embedding is valid
        SET m.taglineEmbedding = emb
        RETURN count(m) as embedded_count
        """
        result = kg.query(
            embedding_query,
            params={
                "apiKey": settings.OPENAI_API_KEY,
                "endpoint": settings.OPENAI_ENDPOINT
            }
        )

        embedded_count = result[0]['embedded_count'] if result and result[0] and result[0]['embedded_count'] is not None else 0
        logger.info(f"✅ Tagline embeddings generation: {embedded_count} movies processed successfully in this run.")

        if embedded_count < candidate_count:
            logger.warning(f"Not all candidates ({candidate_count}) were embedded. {candidate_count - embedded_count} may have failed in GenAI or were filtered. Check Neo4j debug.log.")

    except Exception as e:
        logger.error(f"Error during tagline embedding generation: {e}", exc_info=True)
        logger.error("Ensure OpenAI API key/endpoint are correct and check Neo4j debug.log for GenAI plugin errors.")


### new poster embbedding fucntion poster link is taken for the databse itself


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

def get_poster_urls_batch(kg: Neo4jGraph, skip: int, limit: int) -> List[Dict]:
    """
    Fetches a batch of movie IDs and poster URLs that need embedding.
    Now, `skip` is less critical for iterating through all, but good for resuming.
    The primary control will be fetching until no more are returned.
    """
    # The query itself will only return movies that need embedding.
    # `SKIP` is still useful if you want to resume from a specific point,
    # but the loop termination will be based on an empty result.
    return kg.query("""
        MATCH (m:Movie)
        WHERE m.poster_url IS NOT NULL AND m.poster_embedding IS NULL
        RETURN m.tmdb_id AS id, m.poster_url AS url
        ORDER BY m.tmdb_id // Optional: for consistent processing order if resuming
        SKIP $skip
        LIMIT $limit
    """, params={"skip": skip, "limit": limit})

def generate_batch_image_embeddings_revised(movie_posters: List[Dict]) -> List[Dict]:
    results = []
    for poster in movie_posters:
        movie_id = poster["id"]
        image_url = poster["url"]
        embedding = None
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True).to(device) # Added padding & truncation

            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                embedding = features[0].cpu().tolist()
            logger.debug(f"Successfully generated embedding for movie ID: {movie_id}")
        except requests.exceptions.RequestException as req_e:
            logger.warning(f"⚠️ Movie {movie_id} (URL: {image_url}) failed due to request error: {req_e}")
        except UnidentifiedImageError: # Specific PIL error
             logger.warning(f"⚠️ Movie {movie_id} (URL: {image_url}) failed: Could not identify image file.")
        except Exception as e:
            logger.warning(f"⚠️ Movie {movie_id} (URL: {image_url}) failed during embedding generation: {type(e).__name__} - {e}")
        finally:
            results.append({"movieId": movie_id, "embedding": embedding})

    if device == "cuda":
        torch.cuda.empty_cache()
    return results

# --- (store_embeddings_in_neo4j_optimized - keep as previously optimized) ---



# Uncomment to run the embedding process
# embed_all_posters_in_chunks(kg, batch_size=100, max_movies=10000)

def store_embeddings_in_neo4j_optimized(kg: Neo4jGraph, embeddings_batch: List[Dict]) -> None:
    valid_embeddings = [item for item in embeddings_batch if item.get("embedding")]
    if not valid_embeddings:
        logger.info("No valid embeddings generated in this batch to store.")
        return
    try:
        result = kg.query("""
            UNWIND $batch AS item
            MATCH (m:Movie {tmdb_id: item.movieId})
            SET m.poster_embedding = item.embedding
            RETURN count(m) as updated_count
        """, params={"batch": valid_embeddings})
        updated_count = result[0]['updated_count'] if result else 0
        logger.info(f"Successfully stored {updated_count} poster embeddings in Neo4j for the batch.")
    except Exception as e:
        logger.error(f"Error storing batch embeddings in Neo4j: {e}", exc_info=True)


def embed_all_posters_in_chunks_dynamic(kg: Neo4jGraph, batch_size: int = 50): # Reduced default batch for safety
    """
    Dynamically embeds posters for all movies needing it, without a max_movies limit.
    Uses a resume file to keep track of processed movies by count, not skip position.
    """
    resume_file = "processed_poster_count.txt"
    movies_processed_so_far = 0
    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as f:
                content = f.read().strip()
                if content:
                    movies_processed_so_far = int(content)
            logger.info(f"Resuming poster embedding. Already processed approximately: {movies_processed_so_far} movies.")
        except ValueError:
            logger.warning(f"Could not parse integer from resume file '{resume_file}'. Starting count from 0.")
            movies_processed_so_far = 0 # Reset if file is corrupt
        except Exception as e:
            logger.error(f"Error reading resume file '{resume_file}': {e}. Starting count from 0.")
            movies_processed_so_far = 0

    total_embedded_this_run = 0
    current_skip = movies_processed_so_far # Start skipping based on already processed count

    logger.info(f"Starting poster embedding process. Batch size: {batch_size}. Initial skip: {current_skip}")

    while True:
        logger.info(f"🔁 Fetching next batch of movies to process, skipping first {current_skip} already processed/attempted movies that need embedding.")
        # We still use 'skip' here to effectively page through the remaining movies that need embedding.
        # The query `WHERE ... m.poster_embedding IS NULL` means 'skip' applies to the *subset* of movies needing embedding.
        movies_to_process_this_batch = get_poster_urls_batch(kg, skip=0, limit=batch_size) # Fetch from the start of the *remaining* list

        if not movies_to_process_this_batch:
            logger.info("✅ No more movies found needing poster embeddings. Process complete.")
            break # Exit the loop if no movies are returned

        logger.info(f"Fetched {len(movies_to_process_this_batch)} movie posters for the current batch.")

        embeddings_results = generate_batch_image_embeddings_revised(movies_to_process_this_batch)
        store_embeddings_in_neo4j_optimized(kg, embeddings_results)

        num_in_batch = len(movies_to_process_this_batch)
        movies_processed_so_far += num_in_batch # This count is now the total *attempted* for embedding
        total_embedded_this_run += len([res for res in embeddings_results if res.get("embedding")]) # Count successful ones

        try:
            with open(resume_file, "w") as f:
                # The resume file now stores the count of movies passed to get_poster_urls_batch
                # so that on resume, we effectively 'skip' these.
                # However, since we always query with skip=0 for the *remaining* items,
                # the resume file is more like a progress tracker than a direct skip value
                # for the get_poster_urls_batch function in this dynamic approach.
                # A simpler resume: just log total processed. For robust resume, you'd need to track TMDB_IDs.
                # For this dynamic approach, the 'skip' parameter in get_poster_urls_batch
                # becomes less about resuming a global list and more about simple pagination
                # if the internal Neo4j query optimizer benefits from it.
                # The `WHERE m.poster_embedding IS NULL` is the primary driver for fetching unprocessed items.
                # For now, we'll continue to update `current_skip` as if it's a running total,
                # acknowledging that the `skip=0` in the query handles finding the *next* unprocessed batch.
                # The resume file will reflect the total number of movies *attempted* in previous runs.
                # The important part is that each batch only processes movies whose embedding IS NULL.
                f.write(str(movies_processed_so_far))
        except Exception as e:
            logger.error(f"Error writing to resume file '{resume_file}': {e}")

        logger.info(f"✅ Completed batch. Movies processed in this batch: {num_in_batch}. Total successfully embedded this run: {total_embedded_this_run}. Total attempted across all runs: {movies_processed_so_far}")

        # No explicit `current_skip += batch_size` is needed here because the `get_poster_urls_batch`
        # with `skip=0` and `WHERE m.poster_embedding IS NULL` will always fetch the *next*
        # available batch of unprocessed movies. The `movies_processed_so_far` is more of a
        # progress indicator for the resume file.

    logger.info("🏁 Poster embedding process finished for all available movies.")

#####test function 

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
    #print("Trying to write to the database")
    
    #result=kg.query(
    #"""
    #MATCH(n) 
    #RETURN COUNT(n)
    #"""
    #)
    #print("checking query connection- ", result)
    #print("try to write into the database")
    #writeMovie_to_DB()
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load CLIP (this should be done once globally ideally)
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    logger.info(f"CLIP model loaded to {device}.")

    #embed_all_posters_in_chunks_dynamic(kg, batch_size=100) # Use a smaller batch size for testing
    # titles_for_tagline_test = ["Inception", "The Matrix"]
    # if kg and OPENAI_API_KEY: # Ensure kg and API key are available
    #     generate_tagline_embeddings_test(kg, titles_for_tagline_test)
    # else:
    #     logger.error("Neo4j connection (kg) or OpenAI API Key not available for tagline test.")
    generate_tagline_embeddings() 
if __name__ == '__main__':
    main()



