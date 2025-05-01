import logging
from typing import Dict, Any, List, Tuple, Optional

from graph_db.connection import get_driver, close_driver
from graph_db.query_builder import (
    get_movie_details,
    get_movie_context_for_rag,
    find_movies_by_person,
    execute_query
)
from embeddings.generator import generate_embedding
from embeddings.similarity import find_similar_movies
from llm_integration.chains import create_rag_chain
from utils.fuzzy_matcher import find_best_match
from external_apis.tmdb_client import search_movie_tmdb, get_poster_url

log = logging.getLogger(__name__)

# --- Global instances (consider dependency injection for larger apps) ---
_driver = None
_rag_chain = None

def initialize_services():
    """Initialize Neo4j driver and LLM chain."""
    global _driver, _rag_chain
    if _driver is None:
        try:
            _driver = get_driver()
            log.info("Neo4j Driver obtained for Recommendation Service.")
        except ConnectionError as e:
            log.error(f"Failed to connect to Neo4j: {e}")
            # Decide how to handle this - maybe raise the error, or allow running without DB?
            raise e # Re-raise for now, app can't function without DB
    if _rag_chain is None:
        try:
            _rag_chain = create_rag_chain()
            if _rag_chain:
                log.info("RAG Chain created for Recommendation Service.")
            else:
                log.warning("RAG Chain creation failed (LLM might be disabled or misconfigured). QA features may be limited.")
        except ValueError as e:
            log.error(f"Failed to create RAG chain: {e}")
            # Application might proceed without LLM QA capabilities

def shutdown_services():
    """Clean up resources like the Neo4j driver."""
    global _driver
    log.info("Shutting down recommendation services.")
    if _driver:
        close_driver()
        _driver = None # Reset driver state

def _get_all_movie_titles() -> List[str]:
    """Helper to fetch all movie titles for fuzzy matching. Cache this in production!"""
    # TODO: Implement caching for this list
    if not _driver: return []
    try:
        with _driver.session(database="neo4j") as session:
            result = session.run("MATCH (m:Movie) RETURN m.title AS title")
            titles = [record["title"] for record in result]
            log.info(f"Fetched {len(titles)} movie titles for fuzzy matching.")
            return titles
    except Exception as e:
        log.error(f"Failed to fetch movie titles: {e}")
        return []

def _fetch_movie_details_with_poster(title: str, year: Optional[int] = None) -> Dict[str, Any]:
    """Fetches movie details from Neo4j and adds poster URL from TMDb."""
    movie_data = execute_query(_driver, get_movie_details, title=title)
    if movie_data:
        tmdb_info = search_movie_tmdb(title, year or movie_data.get('released'))
        movie_data['poster_url'] = get_poster_url(tmdb_info)
    return movie_data or {} # Return empty dict if not found


# --- Main Service Function ---

def get_recommendation_or_answer(user_query: str) -> Dict[str, Any]:
    """
    Processes user query to provide movie recommendations or answer questions using RAG.

    Args:
        user_query: The natural language query from the user.

    Returns:
        A dictionary containing the response, type of response, and any relevant data.
        Example: {'type': 'recommendation', 'data': [...], 'explanation': '...'}
                 {'type': 'answer', 'data': '...', 'context_used': True/False}
                 {'type': 'error', 'data': 'Error message'}
    """
    initialize_services() # Ensure services are ready

    if not _driver:
        return {"type": "error", "data": "Database connection is not available."}

    query_lower = user_query.lower()

    # --- Intent Detection (Simple Keyword Based) ---
    # TODO: Replace with a more robust NLU approach if needed

    # 1. Check for Similarity Request
    if "similar to" in query_lower or "like" in query_lower:
        # Extract movie title
        parts = user_query.split("similar to")[-1].split("like")[-1]
        target_movie_title = parts.strip().strip('"').strip("'") # Basic extraction

        # Improve title matching with fuzzy matching
        all_titles = _get_all_movie_titles()
        matched_title = find_best_match(target_movie_title, all_titles)

        if not matched_title:
            return {"type": "error", "data": f"Sorry, I couldn't confidently identify the movie '{target_movie_title}'. Please try again."}

        log.info(f"Similarity request for movie: '{matched_title}' (Original: '{target_movie_title}')")

        # a) Get embedding for the target movie
        movie_details = execute_query(_driver, get_movie_details, title=matched_title)
        if not movie_details or 'plot' not in movie_details:
             return {"type": "error", "data": f"Sorry, I don't have enough information (plot) about '{matched_title}' to find similar movies."}

        target_embedding = generate_embedding(movie_details['plot'])
        if not target_embedding:
             return {"type": "error", "data": f"Could not generate embedding for '{matched_title}'."}

        # b) Perform vector search
        similar_movies_raw = find_similar_movies(_driver, target_embedding, top_k=6) # Get 6 to exclude self

        # c) Format results and add posters
        recommendations = []
        if similar_movies_raw:
             for movie in similar_movies_raw:
                 # Exclude the movie itself from recommendations
                 if movie['title'].lower() != matched_title.lower():
                     details = _fetch_movie_details_with_poster(movie['title'], movie.get('released'))
                     details['similarity_score'] = movie.get('score')
                     recommendations.append(details)

        if not recommendations:
             return {"type": "answer", "data": f"I found '{matched_title}', but couldn't find distinct similar movies based on plot.", "context_used": False}

        # TODO: Optionally add explanation using LLM explanation chain
        explanation = f"Here are some movies similar to '{matched_title}' based on plot similarity:"

        return {
            "type": "recommendation",
            "data": recommendations[:5], # Return top 5 distinct movies
            "explanation": explanation,
            "input_movie": matched_title
         }

    # 2. Check for Question Answering (more general case)
    else:
        log.info(f"Attempting to answer question: '{user_query}' using RAG.")

        if not _rag_chain:
             return {"type": "error", "data": "The question answering service (LLM) is not available."}

        # Try to identify a movie mentioned in the query for context retrieval
        # TODO: Improve entity extraction (e.g., using spaCy or LLM function calling)
        identified_movie_title = None
        all_titles = _get_all_movie_titles() # Fetch titles again (consider caching)
        # Very simple check - see if any movie title is a substring
        for title in all_titles:
             if title.lower() in query_lower:
                 # Use fuzzy matching on the potential match for robustness
                 potential_match = find_best_match(title, [title]) # Check against itself
                 if potential_match:
                     identified_movie_title = potential_match
                     log.info(f"Identified potential movie context: '{identified_movie_title}'")
                     break # Take the first match for simplicity

        # Retrieve context from Graph DB if a movie was identified
        context = None
        if identified_movie_title:
            context = execute_query(_driver, get_movie_context_for_rag, title=identified_movie_title)

        # Prepare input for the RAG chain
        rag_input = {
            "question": user_query,
            "context": context or {} # Provide empty context if no movie identified or found
        }

        # Invoke the RAG chain
        try:
            answer = _rag_chain.invoke(rag_input)
            return {
                "type": "answer",
                "data": answer,
                "context_used": bool(context) # Indicate if graph context was used
            }
        except Exception as e:
            log.error(f"Error invoking RAG chain: {e}")
            return {"type": "error", "data": "Sorry, I encountered an error trying to answer your question."}


# Example of how app.py might call this (without Streamlit context here)
if __name__ == "__main__":
    try:
        initialize_services()

        # Test Case 1: Recommendation
        query1 = "Recommend movies similar to The Matrix"
        result1 = get_recommendation_or_answer(query1)
        print(f"\n--- Result for: '{query1}' ---")
        print(f"Type: {result1.get('type')}")
        if result1.get('type') == 'recommendation':
            print(f"Input Movie: {result1.get('input_movie')}")
            print(f"Explanation: {result1.get('explanation')}")
            for movie in result1.get('data', []):
                print(f"  - {movie.get('title')} ({movie.get('released')}) Score: {movie.get('similarity_score', 'N/A'):.4f} Poster: {movie.get('poster_url')}")
        else:
            print(f"Data: {result1.get('data')}")

        # Test Case 2: RAG Question
        query2 = "Who directed Inception?"
        result2 = get_recommendation_or_answer(query2)
        print(f"\n--- Result for: '{query2}' ---")
        print(f"Type: {result2.get('type')}")
        print(f"Context Used: {result2.get('context_used')}")
        print(f"Answer: {result2.get('data')}")

        # Test Case 3: RAG Question without specific movie context
        query3 = "What are some popular sci-fi movies?"
        result3 = get_recommendation_or_answer(query3)
        print(f"\n--- Result for: '{query3}' ---")
        print(f"Type: {result3.get('type')}")
        print(f"Context Used: {result3.get('context_used')}") # Should likely be False
        print(f"Answer: {result3.get('data')}")

        # Test Case 4: Fuzzy Match Query
        query4 = "Movies like 'incption'"
        result4 = get_recommendation_or_answer(query4)
        print(f"\n--- Result for: '{query4}' ---")
        print(f"Type: {result4.get('type')}")
        if result4.get('type') == 'recommendation':
            print(f"Input Movie: {result4.get('input_movie')}") # Should be 'Inception'
        else:
            print(f"Data: {result4.get('data')}")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
    finally:
        shutdown_services() # Clean up driver connection