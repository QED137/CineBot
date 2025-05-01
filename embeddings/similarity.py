# embeddings/similarity.py

from neo4j import Driver
from typing import List, Dict, Any, Optional
# Ensure these query functions exist in query_builder.py
from graph_db.query_builder import (
    find_movie_by_image_embedding, # <-- Needs to query the image embedding index
    # find_similar_movies_by_embedding, # <-- This one queries the *text* embedding index
    execute_query
    )
import logging

log = logging.getLogger(__name__)

# --- Function for Identifying Movie by Poster Image Embedding ---
def find_movie_by_poster(driver: Driver, query_embedding: List[float], min_score: float = 0.80) -> Optional[Dict[str, Any]]:
    """
    Identifies the most likely movie based on poster image embedding similarity.
    Uses the 'moviePosterIndex' vector index (adjust name if needed).

    Args:
        driver: The Neo4j driver instance.
        query_embedding: The vector embedding of the uploaded image.
        min_score: The minimum cosine similarity score (e.g., 0.80) to consider it a confident match.

    Returns:
        A dictionary representing the matched movie (title, tmdbId, score) if found above threshold, else None.
    """
    if not query_embedding:
        log.warning("Received empty query embedding for poster similarity search.")
        return None
    if not driver:
        log.error("Neo4j driver is not available for poster similarity search.")
        # Or raise an error if driver is essential
        return None

    log.info(f"Performing vector similarity search for poster image (min score: {min_score}).")
    try:
        # Find the single best match using the IMAGE embedding index query
        matches = execute_query(
            driver,
            find_movie_by_image_embedding, # Query function for IMAGE embeddings
            embedding=query_embedding,
            top_k=1 # We only want the single best match for identification
        )

        if matches:
            best_match = matches[0]
            log.info(f"Best poster match candidate: '{best_match.get('title', 'N/A')}' (Score: {best_match.get('score', 0):.4f})")
            # Check if the score meets the confidence threshold
            if best_match.get('score', 0) >= min_score:
                log.info("Match score is above threshold - considering identified.")
                return best_match # Return the dictionary {title: ..., tmdbId: ..., score: ...}
            else:
                log.info("Match score below threshold - not confident.")
                return None
        else:
            log.info("No similar poster found in the database vector index.")
            return None
    except Exception as e:
        # Log the specific error from execute_query or the index call
        log.error(f"An unexpected error occurred during poster similarity search: {e}", exc_info=True)
        return None


# --- Keep Function for Text Embedding Similarity if used elsewhere (e.g., simple recommendations) ---
def find_similar_movies_by_text(driver: Driver, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Finds similar movies based on TEXT plot embedding similarity.
    Uses the 'moviePlotIndex' vector index (adjust name if needed).

    Args:
        driver: The Neo4j driver instance.
        query_embedding: The vector embedding of the query text/plot.
        top_k: The maximum number of similar movies to return.

    Returns:
        A list of dictionaries, each representing a similar movie with title, release year, and score.
        Returns an empty list if an error occurs or no similar movies are found.
    """
    if not query_embedding:
        log.warning("Received empty query embedding for text similarity search.")
        return []
    if not driver:
        log.error("Neo4j driver is not available for text similarity search.")
        return []

    log.info(f"Performing vector similarity search for text embedding (top {top_k}).")
    try:
        # Ensure you use the correct query function for the TEXT index
        # Assuming find_similar_movies_by_embedding queries the text index 'moviePlotIndex'
        # Re-import it explicitly if needed:
        from graph_db.query_builder import find_similar_movies_by_embedding

        similar_movies = execute_query(
            driver,
            find_similar_movies_by_embedding, # Query function for TEXT embeddings
            embedding=query_embedding,
            top_k=top_k
        )
        return similar_movies if similar_movies is not None else []
    except NameError:
         log.error("Query function 'find_similar_movies_by_embedding' for text search not found in query_builder.")
         return []
    except Exception as e:
        log.error(f"An unexpected error occurred during text similarity search: {e}", exc_info=True)
        return []
    