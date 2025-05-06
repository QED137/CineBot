import os
import torch
import requests
from PIL import Image, UnidentifiedImageError # Import UnidentifiedImageError
from typing import List, Dict, Optional
# Assuming your Neo4jGraph import and other initializations are correct
# from langchain_community.graphs import Neo4jGraph (or langchain_neo4j)
# from transformers import CLIPProcessor, CLIPModel
import logging
from langchain_community.graphs import Neo4jGraph

# --- (Keep your existing setup: logging, Neo4jGraph instance 'kg') ---
# --- (Keep your CLIP model and processor loaded globally, and 'device' defined) ---

# Example placeholders if not already defined (ensure these are properly initialized in your script)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# kg = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password") # Replace with your actuals
# clip_model_name = "openai/clip-vit-base-patch32"
# clip_model = CLIPModel.from_pretrained(clip_model_name)
# clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model.to(device)
from config import settings
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)
def connect_neo() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.NEO4J_URI,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database="neo4j"
    )
kg = connect_neo()

def get_specific_movies_for_poster_test(kg: Neo4jGraph, movie_tmdb_ids: List[int]) -> List[Dict]:
    """
    Fetches specific movies by their TMDB IDs for poster embedding testing.
    It will fetch them regardless of whether they already have a poster_embedding,
    allowing for re-testing or testing on already embedded movies.
    """
    if not movie_tmdb_ids:
        logger.warning("No TMDB IDs provided for poster embedding test.")
        return []

    # We fetch movies that have a poster_url.
    # We don't check for 'm.poster_embedding IS NULL' here, so we can re-test.
    return kg.query("""
        MATCH (m:Movie)
        WHERE m.tmdb_id IN $tmdb_ids AND m.poster_url IS NOT NULL AND m.poster_url <> ""
        RETURN m.tmdb_id AS id, m.poster_url AS url, m.title as title
        LIMIT $limit // Limit to the number of IDs provided for safety
    """, params={"tmdb_ids": movie_tmdb_ids, "limit": len(movie_tmdb_ids)})

# --- (Use your existing generate_batch_image_embeddings_revised function) ---
# Make sure it's defined in your script. Here's a recap of a good version:
def generate_batch_image_embeddings_revised(movie_posters: List[Dict]) -> List[Dict]:
    results = []
    for poster_info in movie_posters: # Renamed 'poster' to 'poster_info' for clarity
        movie_id = poster_info["id"]
        image_url = poster_info["url"]
        movie_title = poster_info.get("title", "Unknown Title") # Get title for better logging
        embedding = None
        try:
            logger.debug(f"Processing poster for movie ID: {movie_id} ('{movie_title}'), URL: {image_url}")
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs) # Renamed for clarity
                # Normalize the features
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                embedding = image_features[0].cpu().tolist()
            logger.debug(f"Successfully generated embedding for movie ID: {movie_id} ('{movie_title}')")
        except requests.exceptions.RequestException as req_e:
            logger.warning(f"⚠️ Movie ID {movie_id} ('{movie_title}') URL: {image_url} - Request error: {req_e}")
        except UnidentifiedImageError:
             logger.warning(f"⚠️ Movie ID {movie_id} ('{movie_title}') URL: {image_url} - Could not identify image file (PIL error).")
        except Exception as e:
            logger.warning(f"⚠️ Movie ID {movie_id} ('{movie_title}') URL: {image_url} - Embedding generation error: {type(e).__name__} - {e}")
        finally:
            results.append({"movieId": movie_id, "embedding": embedding})

    if device == "cuda": # Call it once per batch if necessary
        torch.cuda.empty_cache()
    return results

# --- (Use your existing store_embeddings_in_neo4j_optimized function) ---
# Make sure it's defined. Here's a recap:
def store_embeddings_in_neo4j_optimized(kg: Neo4jGraph, embeddings_batch: List[Dict]) -> None:
    valid_embeddings = [item for item in embeddings_batch if item.get("embedding")]
    if not valid_embeddings:
        logger.info("No valid poster embeddings generated in this batch to store.")
        return
    try:
        result = kg.query("""
            UNWIND $batch AS item
            MATCH (m:Movie {tmdb_id: item.movieId})
            SET m.poster_embedding = item.embedding
            RETURN count(m) as updated_count
        """, params={"batch": valid_embeddings})
        updated_count = result[0]['updated_count'] if result and result[0] else 0
        logger.info(f"Successfully stored {updated_count} poster embeddings in Neo4j for the batch.")
    except Exception as e:
        logger.error(f"Error storing batch poster embeddings in Neo4j: {e}", exc_info=True)


def test_poster_embeddings_for_specific_ids(kg: Neo4jGraph, movie_tmdb_ids_to_test: List[int]):
    """
    Tests the poster embedding pipeline for a specific list of TMDB IDs.
    This function will attempt to embed/re-embed the posters for these IDs.
    """
    if not movie_tmdb_ids_to_test:
        logger.info("No movie TMDB IDs provided for poster embedding test.")
        return

    logger.info(f"🧪 Starting poster embedding test for TMDB IDs: {movie_tmdb_ids_to_test}")

    # Optional: Clear existing embeddings if you want a fresh test each time
    # clear_poster_embeddings(kg, movie_tmdb_ids_to_test) # Implement this if needed

    # 1. Get poster URLs for the specific movies
    movies_to_process = get_specific_movies_for_poster_test(kg, movie_tmdb_ids_to_test)

    if not movies_to_process:
        logger.warning(f"Could not find specified movies with poster URLs for TMDB IDs: {movie_tmdb_ids_to_test}. "
                       "Ensure they exist in Neo4j, have a tmdb_id, and a non-empty poster_url.")
        return

    found_ids = [m['id'] for m in movies_to_process]
    logger.info(f"Found {len(movies_to_process)} movies for poster embedding test (IDs: {found_ids}). "
                f"Processing titles: {[m.get('title', 'N/A') for m in movies_to_process]}")


    # 2. Generate embeddings for this small batch
    embeddings_results = generate_batch_image_embeddings_revised(movies_to_process) # list of dicts

    # 3. Store embeddings
    store_embeddings_in_neo4j_optimized(kg, embeddings_results)

    logger.info(f"✅ Poster embedding test pipeline completed for TMDB IDs: {movie_tmdb_ids_to_test}.")

    # 4. Verify (optional, but recommended)
    verify_poster_embeddings_in_db(kg, movie_tmdb_ids_to_test)


def clear_poster_embeddings(kg: Neo4jGraph, movie_tmdb_ids: List[int]):
    """Helper function to remove poster embeddings for specific movies (for re-testing)."""
    if not movie_tmdb_ids: return
    try:
        result = kg.query("""
            MATCH (m:Movie) WHERE m.tmdb_id IN $ids
            REMOVE m.poster_embedding
            RETURN count(m) as cleared_count
        """, params={"ids": movie_tmdb_ids})
        cleared_count = result[0]['cleared_count'] if result and result[0] else 0
        logger.info(f"Cleared existing poster embeddings for {cleared_count} movies (IDs: {movie_tmdb_ids}).")
    except Exception as e:
        logger.error(f"Error clearing poster embeddings: {e}")


def verify_poster_embeddings_in_db(kg: Neo4jGraph, movie_tmdb_ids: List[int]):
    """Helper to verify if poster embeddings were written for given TMDB IDs."""
    if not movie_tmdb_ids: return

    logger.info(f"--- Verifying poster embeddings in DB for TMDB IDs: {movie_tmdb_ids} ---")
    result = kg.query("""
        MATCH (m:Movie) WHERE m.tmdb_id IN $ids
        RETURN m.tmdb_id AS id, m.title AS title,
               m.poster_embedding IS NOT NULL AS hasPosterEmb,
               size(m.poster_embedding) AS posterEmbSize,
               m.poster_url as posterUrl
    """, params={"ids": movie_tmdb_ids})

    if not result:
        logger.warning("No movies found for verification with the given TMDB IDs.")
        return

    for record in result:
        logger.info(
            f"DB Verification - Movie ID {record['id']} ('{record['title']}'): "
            f"Poster Embedded: {record['hasPosterEmb']} (Size: {record['posterEmbSize']}), "
            f"Poster URL: {record['posterUrl']}"
        )
    logger.info("--- Verification complete ---")
def generate_tagline_embeddings_test(kg: Neo4jGraph, movie_titles: List[str]):
    """Generates tagline embeddings for specific movie titles for testing."""
    if not movie_titles:
        logger.warning("No movie titles provided for tagline embedding test.")
        return

    # This query will only run for the specified titles
    # Ensure these movies exist and have taglines in your DB
    kg.query("""
    MATCH (m:Movie) WHERE m.title IN $titles AND m.tagline IS NOT NULL 
    // Optionally add 'AND m.taglineEmbedding IS NULL' if you only want to test fresh ones
    WITH m, genai.vector.encode(
        m.tagline,
        "OpenAI",
        {token: $apiKey, endpoint: $endpoint}
    ) AS embedding
    SET m.taglineEmbedding = embedding
    RETURN m.title AS title, m.taglineEmbedding IS NOT NULL AS embedded
    """, params={"apiKey": OPENAI_API_KEY, "endpoint": OPENAI_ENDPOINT, "titles": movie_titles}) # Assuming OPENAI_API_KEY and OPENAI_ENDPOINT are globally accessible or passed in
    logger.info(f"✅ Tagline embedding test attempted for movies: {', '.join(movie_titles)}.")


# --- Example Usage in your main test section ---
if __name__ == '__main__':
    #Ensure 'kg', 'clip_model', 'clip_processor', 'device' are initialized globally
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    #(Initialize kg, CLIP model etc. here as in your previous example)

    tmdb_ids_for_poster_test = [550, 27205] # Example: Fight Club (550), Batman Begins (27205)
                                    

    if kg: # Check if Neo4j connection is available
        #Optional: Clear embeddings first if you want to ensure a fresh test run
        logger.info(f"Attempting to clear previous poster embeddings for IDs: {tmdb_ids_for_poster_test}")
        clear_poster_embeddings(kg, tmdb_ids_for_poster_test)

        logger.info(f"\n--- Running Poster Embedding Test for TMDB IDs: {tmdb_ids_for_poster_test} ---")
        test_poster_embeddings_for_specific_ids(kg, tmdb_ids_for_poster_test)
    else:
       logger.error("Neo4j connection (kg) not available. Skipping poster embedding test.")