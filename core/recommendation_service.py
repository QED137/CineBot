# core/recommendation_service.py (UPDATED VERSION with Multimodal & Comparison Helper)

import logging
from typing import Dict, Any, List, Optional, Tuple
import io
import pandas as pd # Needed for comparison helper
from neo4j import Driver
# --- Database Imports ---
from graph_db.connection import get_driver, close_driver, _driver # Allow access to _driver for helper
from graph_db.query_builder import (
    get_movie_context_for_rag,
    # get_movie_by_tmdbid, # May not be needed if context has all details
    execute_query
)

# --- Embedding/Similarity Imports ---
# Text embeddings might not be used directly for RAG trigger now
# from embeddings.generator import generate_text_embedding
from embeddings.similarity import find_movie_by_poster
from embeddings.generator import generate_text_embedding # Keep if needed elsewhere

# --- Multimodal Imports ---
from multimodal.image_processor import load_image_from_bytes
from multimodal.vision_model import get_clip_image_embedding

# --- LLM Imports ---
from llm_integration.chains import build_multimodal_rag_messages, invoke_multimodal_llm
# from llm_integration.chains import create_rag_chain # Keep if you want a text-only fallback

# --- Comparison Imports (if comparison logic is here) ---
from comparison.similarity_metrics import (
    get_all_movie_plots_and_embeddings,
    get_tfidf_similarity,
    get_embedding_similarity,
    get_top_n_similar
)


log = logging.getLogger(__name__)

# Simple initialization flag
_services_initialized = False
# Cache for comparison data (avoids refetching from DB constantly)
_comparison_data_cache = None

def initialize_services():
    """Initialize Neo4j driver (LLM loaded on demand)."""
    global _services_initialized
    if _services_initialized:
        return
    if _driver is None: # Check connection module's driver
        try:
            get_driver() # This initializes the driver in connection.py
            log.info("Neo4j Driver obtained for Recommendation Service.")
            _services_initialized = True
        except ConnectionError as e:
            log.error(f"Failed to connect to Neo4j: {e}")
            raise # Re-raise critical error

def shutdown_services():
    """Clean up resources like the Neo4j driver."""
    log.info("Shutting down recommendation services.")
    close_driver() # Use function from connection module
    global _services_initialized, _comparison_data_cache
    _services_initialized = False
    _comparison_data_cache = None # Clear cache on shutdown


# --- Main Multimodal RAG Function ---
def process_input(text_query: Optional[str] = None, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Processes user text query and/or image upload for multimodal RAG.

    Returns:
        A dictionary containing the response: {'answer': '...', 'identified_movie': '...', 'error': '...'}
    """
    initialize_services() # Ensure driver is ready

    if not _driver:
        return {"error": "Database connection is not available."}

    identified_movie_title = None
    graph_context = None
    final_response = None
    response_dict = {}
    processed_image = False # Flag to track if image was processed

    # --- 1. Process Image Input (if provided) ---
    if image_bytes:
        processed_image = True # Mark that we attempted image processing
        log.info("Processing uploaded image...")
        try:
            pil_image = load_image_from_bytes(image_bytes)
            query_embedding = get_clip_image_embedding(pil_image)

            if query_embedding:
                # Try to identify movie by poster similarity
                # Use a reasonable similarity threshold (e.g., 0.80 - tune this)
                matched_movie = find_movie_by_poster(_driver, query_embedding, min_score=0.80)
                if matched_movie:
                    identified_movie_title = matched_movie.get('title')
                    response_dict['identified_movie'] = identified_movie_title
                    log.info(f"Image identified as likely poster for: {identified_movie_title}")
                    # Retrieve graph context for the identified movie
                    graph_context = execute_query(_driver, get_movie_context_for_rag, title=identified_movie_title)
                    if not graph_context:
                         log.warning(f"Identified movie '{identified_movie_title}' but failed to retrieve its context from graph.")
                         # Keep identified_movie_title but context is None
                else:
                    log.info("Uploaded image did not confidently match any known posters.")
                    response_dict['image_analysis'] = "Image uploaded, but couldn't identify a specific movie poster from the database."
            else:
                log.warning("Could not generate embedding for the uploaded image.")
                response_dict['error'] = "Failed to process the uploaded image."
                # Optionally: Could try image captioning here if poster ID fails

        except Exception as e:
            log.error(f"Error during image processing or identification: {e}", exc_info=True)
            # Don't immediately return error, allow text processing if available
            response_dict['image_processing_error'] = f"Error processing image: {e}"

    # --- 2. Process Text Input & Retrieve Context (if no image context or text is primary) ---
    # If an image was identified and context found, we prioritize its context.
    # If no image, or image not identified/no context found, use text query to find context.
    if not graph_context and text_query:
        log.info(f"Processing text query: {text_query}")
        # Simple context retrieval: Check if a known movie title is in the query
        # TODO: Improve entity extraction for robust movie title identification from text
        # For demo, let's assume user query might *be* the title for simplicity
        potential_context = execute_query(_driver, get_movie_context_for_rag, title=text_query.strip())
        if potential_context and potential_context.get('title'):
             identified_movie_title = potential_context.get('title') # Use title from DB
             graph_context = potential_context
             log.info(f"Found graph context for movie mentioned in text: {identified_movie_title}")
             # Don't overwrite response_dict['identified_movie'] if image was identified earlier
             if 'identified_movie' not in response_dict:
                response_dict['identified_movie'] = identified_movie_title

    # --- 3. Build Multimodal Message & Invoke LLM ---
    # Determine the primary question/task for the LLM
    if not text_query and identified_movie_title:
        llm_question = f"Tell me about the movie '{identified_movie_title}' shown in the uploaded image."
    elif not text_query and processed_image and 'image_analysis' in response_dict:
         llm_question = "Describe the scene or poster shown in the uploaded image. What movie might it be related to?"
    elif not text_query and processed_image and 'image_processing_error' in response_dict:
         llm_question = "There was an error processing the uploaded image. Please describe any issues." # Fallback
    elif text_query:
        llm_question = text_query
    else:
        # No input provided that could be processed
        if 'image_processing_error' in response_dict: # Prioritize image error if it occurred
             return {"error": response_dict['image_processing_error']}
        return {"error": "No valid query or image provided."}

    try:
        # Only pass image_bytes if processing was attempted (even if identification failed)
        messages = build_multimodal_rag_messages(llm_question, graph_context, image_bytes if processed_image else None)
        final_response = invoke_multimodal_llm(messages)
        response_dict['answer'] = final_response
    except ValueError as e: # Catch LLM init errors
         log.error(f"LLM Error: {e}")
         response_dict['error'] = f"LLM Error: {e}"
    except Exception as e:
         log.error(f"Failed to get response from LLM: {e}", exc_info=True)
         response_dict['error'] = "Error communicating with the AI model."

    # Clean up potential error keys if answer was successful
    if 'answer' in response_dict:
         response_dict.pop('image_processing_error', None)
         response_dict.pop('error', None)
    elif 'error' not in response_dict and 'image_processing_error' in response_dict:
        # If only image processing failed but we didn't generate an answer
        response_dict['error'] = response_dict['image_processing_error']


    return response_dict

# --- Helper Function for Comparison UI ---
def get_all_movie_titles_for_comparison(db_driver: Optional[Driver]) -> List[str]:
    """Fetches a distinct list of movie titles for the comparison dropdown."""
    if not db_driver:
        log.warning("DB driver not available for fetching movie titles.")
        return ["Error: DB Driver unavailable"]
    try:
        # Efficient query to get just distinct titles
        cypher = "MATCH (m:Movie) WHERE m.title IS NOT NULL RETURN COLLECT(DISTINCT m.title) AS titles"
        result = execute_query(db_driver, lambda tx: tx.run(cypher).single())
        if result and result['titles']:
             return sorted(result['titles'])
        else:
             return ["Error: No titles found"]
    except Exception as e:
        log.error(f"Failed to fetch distinct movie titles: {e}")
        return [f"Error fetching titles: {e}"]


# --- Comparison Function ---
def get_comparison_results(target_movie_title: str) -> Dict[str, Any]:
    """
    Performs similarity comparisons using different methods for a target movie.
    Uses a simple cache (_comparison_data_cache).
    """
    global _comparison_data_cache
    initialize_services()
    if not _driver:
        return {"error": "Database connection is not available."}

    # --- 1. Load or Get Cached Data ---
    # Consider adding a refresh mechanism or time-based invalidation for cache
    if _comparison_data_cache is None or _comparison_data_cache.empty:
        log.info("Comparison cache miss or empty. Fetching data from Neo4j...")
        # Use the function from comparison_metrics or define fetching here
        _comparison_data_cache = get_all_movie_plots_and_embeddings(_driver)
        if _comparison_data_cache.empty:
            log.error("Failed to load data needed for comparison from database.")
            # Clear cache in case of partial failure?
            _comparison_data_cache = None
            return {"error": "Could not load data needed for comparison."}
        log.info("Comparison data loaded and cached.")
    else:
         log.info("Using cached comparison data.")

    movie_df = _comparison_data_cache

    # --- 2. Find Target Movie Data ---
    target_data = movie_df[movie_df['title'].str.lower() == target_movie_title.lower()]
    if target_data.empty:
        # Optional: Add fuzzy matching here if needed
        return {"error": f"Movie '{target_movie_title}' not found in the comparison dataset."}

    target_row = target_data.iloc[0]
    target_plot = target_row['plot']
    # Ensure embeddings are actually lists of numbers, handle potential None/empty
    target_text_embedding = target_row['textEmbedding'] if isinstance(target_row['textEmbedding'], list) else None
    target_image_embedding = target_row['imageEmbedding'] if isinstance(target_row['imageEmbedding'], list) else None


    # --- 3. Calculate Similarities ---
    results = {"target_movie": target_movie_title}
    error_messages = []

    # TF-IDF Similarity
    log.info(f"Calculating TF-IDF similarity for '{target_movie_title}'...")
    try:
        tfidf_results_raw = get_tfidf_similarity(target_plot, movie_df)
        if tfidf_results_raw:
            results['tfidf'] = get_top_n_similar(tfidf_results_raw, target_movie_title, n=5)
        else: error_messages.append("TF-IDF similarity failed.")
    except Exception as e:
         log.error(f"TF-IDF calculation failed: {e}", exc_info=True)
         error_messages.append("TF-IDF calculation error.")

    # Text Embedding Similarity
    log.info(f"Calculating Text Embedding similarity for '{target_movie_title}'...")
    if target_text_embedding:
         try:
             text_embeddings_list = movie_df['textEmbedding'].tolist()
             titles_list = movie_df['title'].tolist()
             # Filter out potential None values if necessary
             valid_text_data = [(emb, title) for emb, title in zip(text_embeddings_list, titles_list) if isinstance(emb, list) and emb]
             if valid_text_data:
                 valid_text_embeddings = [item[0] for item in valid_text_data]
                 valid_titles = [item[1] for item in valid_text_data]
                 text_emb_results_raw = get_embedding_similarity(target_text_embedding, valid_text_embeddings, valid_titles)
                 if text_emb_results_raw:
                     results['text_embedding'] = get_top_n_similar(text_emb_results_raw, target_movie_title, n=5)
                 else: error_messages.append("Text Embedding similarity failed.")
             else: error_messages.append("No valid text embeddings found for comparison.")
         except Exception as e:
              log.error(f"Text Embedding similarity calculation failed: {e}", exc_info=True)
              error_messages.append("Text Embedding similarity calculation error.")
    else: error_messages.append("Target movie missing text embedding.")


    # Image Embedding Similarity
    log.info(f"Calculating Image Embedding similarity for '{target_movie_title}'...")
    if target_image_embedding:
         try:
             image_embeddings_list = movie_df['imageEmbedding'].tolist()
             titles_list = movie_df['title'].tolist()
             # Filter out potential None/empty values
             valid_image_data = [(emb, title) for emb, title in zip(image_embeddings_list, titles_list) if isinstance(emb, list) and emb]
             if valid_image_data:
                 valid_image_embeddings = [item[0] for item in valid_image_data]
                 valid_titles = [item[1] for item in valid_image_data]
                 image_emb_results_raw = get_embedding_similarity(target_image_embedding, valid_image_embeddings, valid_titles)
                 if image_emb_results_raw:
                     results['image_embedding'] = get_top_n_similar(image_emb_results_raw, target_movie_title, n=5)
                 else: error_messages.append("Image Embedding similarity calculation failed.")
             else: error_messages.append("No valid image embeddings found in dataset for comparison.")
         except Exception as e:
              log.error(f"Image Embedding similarity calculation failed: {e}", exc_info=True)
              error_messages.append("Image Embedding similarity calculation error.")
    else: error_messages.append("Target movie missing image embedding.")

    if error_messages:
        results['errors'] = error_messages

    return results


# Keep the old text-only function if you need a fallback or specific text logic
# def get_recommendation_or_answer(user_query: str) -> Dict[str, Any]:
#     # ... (original text-only logic from the file you provided) ...
#     # This would use _rag_chain = create_rag_chain() etc.
#     pass

# Note: The __main__ block from your original file is removed as this
# module is intended to be imported, not run directly usually.
# Testing should be done via dedicated test scripts or by running app.py.