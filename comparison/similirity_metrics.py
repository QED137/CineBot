import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
from neo4j import Driver
from graph_db.query_builder import execute_query

log = logging.getLogger(__name__)

# --- Data Fetching ---
def get_all_movie_plots_and_embeddings(driver: Driver) -> pd.DataFrame:
    """Fetches titles, plots, text embeddings, and poster embeddings for all movies."""
    log.info("Fetching movie data for comparison (titles, plots, embeddings)...")
    cypher = """
    MATCH (m:Movie)
    WHERE m.plot IS NOT NULL AND m.plot <> "" AND m.plotEmbedding IS NOT NULL AND m.posterEmbedding IS NOT NULL
    RETURN m.title AS title,
           m.plot AS plot,
           m.plotEmbedding AS textEmbedding,
           m.posterEmbedding AS imageEmbedding
    """
    try:
        results = execute_query(driver, lambda tx: tx.run(cypher).data())
        if not results:
            log.warning("No movie data found for comparison.")
            return pd.DataFrame()
        df = pd.DataFrame(results)
        log.info(f"Fetched data for {len(df)} movies.")
        # Convert embeddings back to numpy arrays if needed, though lists might work directly
        # df['textEmbedding'] = df['textEmbedding'].apply(np.array)
        # df['imageEmbedding'] = df['imageEmbedding'].apply(np.array)
        return df
    except Exception as e:
        log.error(f"Failed to fetch movie data for comparison: {e}")
        return pd.DataFrame()

# --- TF-IDF Calculation ---
tfidf_vectorizer = None
tfidf_matrix = None
tfidf_titles = None

def compute_tfidf(plots: pd.Series):
    """Computes TF-IDF matrix for given plots."""
    global tfidf_vectorizer, tfidf_matrix, tfidf_titles
    log.info("Calculating TF-IDF matrix...")
    try:
        # Use stop words, limit features if dataset is large
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(plots)
        log.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_vectorizer, tfidf_matrix
    except Exception as e:
        log.error(f"Failed to compute TF-IDF: {e}")
        return None, None

# --- Similarity Calculations ---
def get_tfidf_similarity(target_plot: str, movie_df: pd.DataFrame) -> Optional[List[Tuple[str, float]]]:
    """Calculates cosine similarity based on TF-IDF."""
    global tfidf_vectorizer, tfidf_matrix
    if tfidf_vectorizer is None or tfidf_matrix is None:
        # Compute TF-IDF if not already done (pass all plots from df)
        tfidf_vectorizer, tfidf_matrix = compute_tfidf(movie_df['plot'])
        if tfidf_vectorizer is None: return None # Handle computation error

    try:
        target_vector = tfidf_vectorizer.transform([target_plot])
        cosine_similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()
        # Get top N matches (excluding self if present)
        # Combine titles with scores
        results = list(zip(movie_df['title'], cosine_similarities))
        return results
    except Exception as e:
        log.error(f"Error calculating TF-IDF similarity: {e}")
        return None

def get_embedding_similarity(target_embedding: List[float], embeddings: List[List[float]], titles: List[str]) -> Optional[List[Tuple[str, float]]]:
    """Calculates cosine similarity for pre-computed embeddings (text or image)."""
    if not target_embedding or not embeddings:
        return None
    try:
        # Ensure embeddings are numpy arrays for cosine_similarity
        import numpy as np
        target_vec = np.array(target_embedding).reshape(1, -1)
        embedding_matrix = np.array(embeddings)

        # Check dimensions match
        if target_vec.shape[1] != embedding_matrix.shape[1]:
            log.error(f"Embedding dimension mismatch: Target {target_vec.shape[1]}, Matrix {embedding_matrix.shape[1]}")
            return None

        cosine_similarities = cosine_similarity(target_vec, embedding_matrix).flatten()
        results = list(zip(titles, cosine_similarities))
        return results
    except ValueError as ve: # Handle potential numpy array conversion errors
         log.error(f"Error during embedding similarity calculation (check embedding lists): {ve}")
         return None
    except Exception as e:
        log.error(f"Error calculating embedding similarity: {e}")
        return None

def get_top_n_similar(results: List[Tuple[str, float]], target_title: str, n: int = 5) -> List[Dict[str, Any]]:
    """Sorts similarity results and returns top N, excluding the target movie."""
    if not results:
        return []
    # Sort by score descending
    results.sort(key=lambda item: item[1], reverse=True)
    # Filter out self and take top N
    top_n = []
    for title, score in results:
        if title.lower() != target_title.lower(): # Case-insensitive comparison
            top_n.append({"title": title, "score": score})
            if len(top_n) >= n:
                break
    return top_n