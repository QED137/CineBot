from neo4j import Driver
from typing import List, Dict, Any
from graph_db.query_builder import find_similar_movies_by_embedding, execute_query
import logging

log = logging.getLogger(__name__)

def find_similar_movies(driver: Driver, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Finds similar movies based on embedding similarity using Neo4j vector index.

    Args:
        driver: The Neo4j driver instance.
        query_embedding: The vector embedding to search for.
        top_k: The maximum number of similar movies to return.

    Returns:
        A list of dictionaries, each representing a similar movie with title, release year, and score.
        Returns an empty list if an error occurs or no similar movies are found.
    """
    if not query_embedding:
        log.warning("Received empty query embedding for similarity search.")
        return []
    if not driver:
        log.error("Neo4j driver is not available for similarity search.")
        return []

    log.info(f"Performing vector similarity search for top {top_k} movies.")
    try:
        similar_movies = execute_query(
            driver,
            find_similar_movies_by_embedding,
            embedding=query_embedding,
            top_k=top_k
        )
        return similar_movies if similar_movies is not None else []
    except Exception as e:
        # execute_query already logs the error
        log.error(f"An unexpected error occurred during similarity search: {e}")
        return []

# Note: If you were doing in-memory similarity (e.g., with cached embeddings)
# you would use libraries like numpy or scikit-learn here.
# Example (Conceptual - requires loading all embeddings into memory):
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
#
# all_movie_embeddings = {} # Dict mapping title -> embedding (np.array)
#
# def find_similar_in_memory(query_embedding: np.ndarray, top_k: int = 5):
#     if not all_movie_embeddings:
#         # Load or cache embeddings first
#         return []
#     titles = list(all_movie_embeddings.keys())
#     embeddings_matrix = np.array(list(all_movie_embeddings.values()))
#     similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings_matrix)[0]
#     # Get indices of top_k highest similarities
#     top_indices = np.argsort(similarities)[::-1][:top_k]
#     results = [(titles[i], similarities[i]) for i in top_indices]
#     return results