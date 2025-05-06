import time
import logging
# Assuming 'settings' is your config object and 'kg' is your Neo4jGraph instance
# from config import settings
# from langchain_community.graphs import Neo4jGraph

from config import settings
from langchain_community.graphs import Neo4jGraph
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from urllib.parse import quote
import os
from typing import List, Dict, Optional


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
def create_vector_indexes() -> None:
    
    kg.query("""
    CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
    FOR (m:Movie) ON (m.taglineEmbedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
    """)
def get_candidate_movies_for_tagline_embedding_batch(
    kg_conn: Neo4jGraph,
    limit: int
) -> List[int]: # Returns a list of tmdb_ids
    """
    Fetches a batch of tmdb_ids for movies that need tagline embedding.
    """
    query = """
    MATCH (m:Movie)
    WHERE m.tagline IS NOT NULL
      AND m.tagline <> ""
      AND m.taglineEmbedding IS NULL
    RETURN m.tmdb_id AS id
    LIMIT $limit
    """
    try:
        results = kg_conn.query(query, params={"limit": limit})
        return [r['id'] for r in results] if results else []
    except Exception as e:
        logger.error(f"Error fetching candidate movie IDs for tagline batch: {e}", exc_info=True)
        return []


def embed_taglines_for_specific_ids(
    kg_conn: Neo4jGraph,
    movie_ids_batch: List[int],
    api_key: str,
    api_endpoint: str
) -> int: # Returns the count of successfully embedded movies in this batch
    """
    Generates and sets tagline embeddings for a specific list of movie TMDB IDs.
    """
    if not movie_ids_batch:
        return 0

    embedding_query = """
    MATCH (m:Movie)
    WHERE m.tmdb_id IN $ids // Process only movies in the provided ID batch
      AND m.tagline IS NOT NULL // Re-check conditions for safety, though ideally already filtered
      AND m.tagline <> ""
      AND m.taglineEmbedding IS NULL
    WITH m, genai.vector.encode(
        m.tagline,
        "OpenAI",
        {
            token: $apiKey,
            endpoint: $endpoint
        }
    ) AS emb
    WHERE emb IS NOT NULL AND size(emb) > 0 // Ensure embedding is valid
    SET m.taglineEmbedding = emb
    RETURN count(m) as embedded_count_in_batch
    """
    try:
        logger.debug(f"Embedding taglines for {len(movie_ids_batch)} movies (IDs: {movie_ids_batch[:5]}...).")
        result = kg_conn.query(
            embedding_query,
            params={
                "ids": movie_ids_batch,
                "apiKey": api_key,
                "endpoint": api_endpoint
            }
        )
        embedded_count = result[0]['embedded_count_in_batch'] if result and result[0] and result[0]['embedded_count_in_batch'] is not None else 0
        logger.info(f"Successfully set tagline embeddings for {embedded_count} out of {len(movie_ids_batch)} movies in the current batch.")
        if embedded_count < len(movie_ids_batch):
            logger.warning(f"{len(movie_ids_batch) - embedded_count} movies in the batch did not get embeddings. Check Neo4j debug.log for potential individual GenAI errors.")
        return embedded_count
    except Exception as e:
        logger.error(f"Error during batched tagline embedding Cypher query for IDs {movie_ids_batch[:5]}...: {e}", exc_info=True)
        logger.error("PLEASE CHECK THE NEO4J DEBUG.LOG for specific GenAI plugin errors.")
        return 0


def generate_taglines_in_batches_with_delay(
    kg_conn: Neo4jGraph,
    batch_size: int = 100, # Adjust as needed, might be higher than poster batch size
    delay_between_batches: int = 5 # Adjust based on API rate limits
):
    """
    Generates tagline embeddings for all eligible movies in batches with delays.
    """
    if not kg_conn:
        logger.error("Neo4j connection (kg_conn) not provided. Skipping tagline embedding.")
        return
    if not hasattr(settings, 'OPENAI_API_KEY') or not settings.OPENAI_API_KEY:
        logger.error("settings.OPENAI_API_KEY is not set. Skipping tagline embedding.")
        return
    api_key = settings.OPENAI_API_KEY
    api_endpoint = getattr(settings, 'OPENAI_ENDPOINT', None) # Safely get endpoint
    if not api_endpoint:
        logger.warning("settings.OPENAI_ENDPOINT is not set. GenAI plugin might use a default or fail if required.")


    logger.info(f"Starting batched tagline embedding process. Batch size: {batch_size}, Delay: {delay_between_batches}s.")
    total_movies_processed_successfully = 0
    batch_num = 0

    while True:
        batch_num += 1
        logger.info(f"--- Tagline Batch #{batch_num} ---")

        candidate_ids = get_candidate_movies_for_tagline_embedding_batch(kg_conn, limit=batch_size)

        if not candidate_ids:
            logger.info("✅ No more movies found requiring tagline embeddings. Process complete.")
            break

        logger.info(f"Fetched {len(candidate_ids)} movie IDs for tagline embedding batch #{batch_num}.")

        successfully_embedded_in_batch = embed_taglines_for_specific_ids(
            kg_conn,
            candidate_ids,
            api_key,
            api_endpoint
        )
        total_movies_processed_successfully += successfully_embedded_in_batch

        logger.info(f"✅ Tagline Batch #{batch_num} completed. Successfully embedded: {successfully_embedded_in_batch}. "
                    f"Total successfully embedded so far: {total_movies_processed_successfully}.")

        # No resume file needed here in the same way as posters, because each batch
        # re-queries for the *next available* unprocessed movies. If the script
        # stops, the next run will pick up where it left off due to the
        # `m.taglineEmbedding IS NULL` condition.

        logger.info(f"Waiting for {delay_between_batches} seconds before next tagline batch...")
        time.sleep(delay_between_batches)

    logger.info(f"🏁 Batched tagline embedding process finished. Total movies successfully embedded: {total_movies_processed_successfully}.")


# --- In your main_workflow() or equivalent ---
# --- In your main_workflow() or equivalent ---
def main():
    # Ensure kg is initialized and working
    if not kg:
        logger.error("Neo4j connection 'kg' is not initialized. Exiting.")
        return
    try:
        kg.query("RETURN 1") # Test connection
    except Exception as e:
        logger.error(f"Neo4j connection test failed: {e}. Exiting.")
        return

    logger.info("--- Ensuring Tagline Vector Index Exists ---")
    create_vector_indexes() # Call this to ensure index is there

    logger.info("\n--- Starting Batched Tagline Embedding Process ---")
    generate_taglines_in_batches_with_delay(
        kg_conn=kg,
        batch_size=100,
        delay_between_batches=5
    )
    logger.info("--- Tagline Embedding Workflow Completed ---")

if __name__ == '__main__':
    # Optional: Add any one-time setup here if not done globally,
    # but your current global setup for logging and kg is good.
    main() # Call the main function to start the process
  