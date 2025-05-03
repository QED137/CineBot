from neo4j import Driver, Session, Transaction
from typing import List, Dict, Any, Optional
import logging
from graph_db.connection import get_driver, close_driver
from  config import settings
log = logging.getLogger(__name__)

# --- Basic Node/Relationship Retrieval ---

def get_movie_details(tx: Transaction, title: str) -> Optional[Dict[str, Any]]:
    """Fetches details for a specific movie."""
    cypher = """
    MATCH (m:Movie {title: $title})
    RETURN m.title AS title, m.released AS released, m.tagline AS tagline, m.plot AS plot
    LIMIT 1
    """
    result = tx.run(cypher, title=title)
    record = result.single()
    return record.data() if record else None

def get_movie_context_for_rag(tx: Transaction, title: str) -> Optional[Dict[str, Any]]:
    """Fetches structured context about a movie for RAG."""
    # This query gathers the movie's direct connections
    cypher = """
    MATCH (m:Movie {title: $title})
    OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
    OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
    RETURN
        m.title AS title,
        m.released AS released,
        m.tagline AS tagline,
        m.plot AS plot,
        collect(DISTINCT d.name) AS directors,
        collect(DISTINCT a.name) AS actors,
        collect(DISTINCT g.name) AS genres
    LIMIT 1
    """
    result = tx.run(cypher, title=title)
    record = result.single()
    if not record or not record['title']: # Check if movie was found
        return None

    # Format the context nicely for the LLM
    context = {
        "title": record["title"],
        "released": record["released"],
        "tagline": record["tagline"],
        "plot": record["plot"], # Keep plot separate or shorten if too long for prompt
        "directors": record["directors"] if record["directors"] else ["N/A"],
        "actors": record["actors"] if record["actors"] else ["N/A"],
        "genres": record["genres"] if record["genres"] else ["N/A"]
    }
    return context


def find_movies_by_person(tx: Transaction, person_name: str) -> List[Dict[str, Any]]:
    """Finds movies associated with a person (actor or director)."""
    cypher = """
    MATCH (p:Person {name: $name})-[r:ACTED_IN|DIRECTED]->(m:Movie)
    RETURN m.title AS title, m.released AS released, type(r) AS role
    ORDER BY m.released DESC
    """
    result = tx.run(cypher, name=person_name)
    return [record.data() for record in result]

def find_movies_by_genre(tx: Transaction, genre_name: str) -> List[Dict[str, Any]]:
    """Finds movies belonging to a specific genre."""
    cypher = """
    MATCH (g:Genre {name: $genre})<-[:HAS_GENRE]-(m:Movie)
    RETURN m.title AS title, m.released AS released
    ORDER BY m.released DESC
    LIMIT 20
    """
    result = tx.run(cypher, genre=genre_name)
    return [record.data() for record in result]

# --- Vector Similarity Search ---

def find_similar_movies_by_embedding(tx: Transaction, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Finds movies similar to the given embedding using a vector index."""
    # Assumes a vector index named 'moviePlotIndex' exists on Movie.plotEmbedding
    # Adjust index name and property key ('plotEmbedding') as needed
    cypher = """
    CALL db.index.vector.queryNodes('moviePlotIndex', $top_k, $embedding)
    YIELD node AS similarMovie, score
    RETURN similarMovie.title AS title, similarMovie.released AS released, score
    ORDER BY score DESC
    """
    result = tx.run(cypher, embedding=embedding, top_k=top_k)
    return [record.data() for record in result]

# --- Helper to execute queries ---

def execute_query(driver: Driver, query_func, **kwargs) -> Any:
    """Helper function to execute a query function within a session."""
    try:
        with driver.session(database="neo4j") as session: # Specify database if needed
            return session.execute_read(query_func, **kwargs)
            # Use execute_write for CUD operations (CREATE, UPDATE, DELETE)
    except Exception as e:
        log.error(f"Error executing query {query_func.__name__}: {e}")
        # Depending on the use case, you might want to return None, an empty list, or re-raise
        return None # Or return [] / raise e

# --- Function to find movie by IMAGE embedding ---
def find_movie_by_image_embedding(tx: Transaction, embedding: List[float], top_k: int = 1) -> List[Dict[str, Any]]:
    """Finds movies based on image embedding similarity using a vector index."""
    # Assumes a vector index named 'moviePosterIndex' exists on Movie.posterEmbedding
    # Adjust index name and property key as needed
    cypher = """
    CALL db.index.vector.queryNodes('moviePosterIndex', $top_k, $embedding)
    YIELD node AS matchedMovie, score
    RETURN matchedMovie.title AS title,
           matchedMovie.tmdbId AS tmdbId, // Return tmdbId for potential lookups
           score
    ORDER BY score DESC
    """
    result = tx.run(cypher, embedding=embedding, top_k=top_k)
    return [record.data() for record in result]

# --- Optional: Function to get movie details by tmdbId ---
def get_movie_by_tmdbid(tx: Transaction, tmdb_id: int) -> Optional[Dict[str, Any]]:
    """Fetches minimal details for a movie by its TMDb ID."""
    cypher = """
    MATCH (m:Movie {tmdbId: $tmdb_id})
    RETURN m.title AS title, m.released AS released
    LIMIT 1
    """
    result = tx.run(cypher, tmdb_id=tmdb_id)
    record = result.single()
    return record.data() if record else None
def create_tagline_embeddings(movies):
    cypher = """
    CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
    FOR (m:Movie) ON (m.taglineEmbedding) 
    OPTIONS { indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
        }}"""
        
    

# Example usage (can be tested independently)
if __name__ == '__main__':
    

    driver = get_driver()
    try:
        # Example: Get details for a movie
        movie_title = "The Replacements"
        details = execute_query(driver, get_movie_details, title=movie_title)
        if details:
            print(f"\nDetails for {movie_title}: {details}")
        else:
            print(f"\nMovie '{movie_title}' not found.")

         # Example: Get RAG context
        context = execute_query(driver, get_movie_context_for_rag, title=movie_title)
        if context:
            print(f"\nRAG Context for {movie_title}:\n{context}")

        # Example: Find movies by person
        person = "Keanu Reeves"
        movies = execute_query(driver, find_movies_by_person, person_name=person)
        if movies:
            print(f"\nMovies starring/directed by {person}:")
            for movie in movies[:5]: # Print first 5
                print(f"- {movie['title']} ({movie['released']}) - Role: {movie['role']}")

        # Example: Vector search (requires index and embeddings in DB)
        # dummy_embedding = [0.1] * 1536 # Replace with actual embedding size/values
        # similar = execute_query(driver, find_similar_movies_by_embedding, embedding=dummy_embedding, top_k=3)
        # if similar:
        #     print(f"\nMovies similar to embedding:")
        #     for movie in similar:
        #         print(f"- {movie['title']} ({movie['released']}) - Score: {movie['score']:.4f}")

    finally:
        close_driver()