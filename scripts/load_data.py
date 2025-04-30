import argparse
import logging
import time
import requests # To call TMDb API
import pandas as pd # Still useful for structuring data before loading
from typing import List, Dict, Any, Optional
from neo4j import Driver, Transaction

# Adjust imports based on your project structure
from graph_db.connection import get_driver, close_driver
from embeddings.generator import generate_embedding, get_embedding_model
from config import settings # Ensure settings loads TMDB_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Constants ---

# smaller sizes
global MAX_TMDB_PAGES, NEO4J_BATCH_SIZE
MAX_TMDB_PAGES = args.max_pages
NEO4J_BATCH_SIZE = args.batch_size
NEO4J_BATCH_SIZE = 100
TMDB_API_KEY = settings.TMDB_API_KEY
TMDB_BASE_URL = "https://api.themoviedb.org/3"
# Number of TMDb pages to fetch (each page has 20 movies typically)
# Number of movies (e.g., 50 pages * 20 movies/page = 1000 movies)
MAX_TMDB_PAGES = 50
# Delay between API calls to respect TMDb rate limits
API_DELAY_SECONDS = 0.3

# --- Cypher Queries (Same as before, ensure index dimension matches model) ---

# Create constraints for uniqueness (improves performance and data integrity)
CREATE_CONSTRAINTS_CYPHER = [
    "CREATE CONSTRAINT movieTitle IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE;",
    "CREATE CONSTRAINT personName IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE;",
    "CREATE CONSTRAINT genreName IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;",
]

# Cypher query to batch load movies and relationships
LOAD_MOVIES_CYPHER = """
UNWIND $batch AS movie_data

// Merge Movie node
MERGE (m:Movie {title: movie_data.title})
ON CREATE SET
    m.released = movie_data.released,
    m.release_date = movie_data.release_date, // Store full date if available
    m.tagline = movie_data.tagline,
    m.plot = movie_data.plot,
    m.poster_path = movie_data.poster_path, // Store only the path from TMDb
    m.tmdbId = movie_data.tmdbId,
    m.vote_average = movie_data.vote_average,
    m.vote_count = movie_data.vote_count,
    m.plotEmbedding = movie_data.plotEmbedding // Store the generated embedding
ON MATCH SET // Update existing movies with potentially new data
    m.released = movie_data.released,
    m.release_date = movie_data.release_date,
    m.tagline = movie_data.tagline,
    m.plot = movie_data.plot,
    m.poster_path = movie_data.poster_path,
    m.tmdbId = movie_data.tmdbId,
    m.vote_average = movie_data.vote_average,
    m.vote_count = movie_data.vote_count,
    m.plotEmbedding = movie_data.plotEmbedding // Update embedding too

// Merge Director node and relationship (handle potential missing director)
// Assuming movie_data.director is a list (can be empty)
FOREACH (director_name IN movie_data.directors |
    MERGE (d:Person {name: director_name})
    MERGE (d)-[:DIRECTED]->(m)
)

// Merge Genre nodes and relationships
FOREACH (genre_name IN movie_data.genres |
    MERGE (g:Genre {name: genre_name})
    MERGE (m)-[:HAS_GENRE]->(g)
)

// Merge Actor nodes and relationships (limit number of actors?)
FOREACH (actor_name IN movie_data.actors |
    MERGE (a:Person {name: actor_name})
    MERGE (a)-[:ACTED_IN]->(m)
)
"""

# Cypher queries to create indexes (including vector index)
# Function to dynamically generate the vector index query
def get_create_indexes_cypher(embedding_dim: int) -> List[str]:
    return [
        "CREATE INDEX movieReleased IF NOT EXISTS FOR (m:Movie) ON (m.released);",
        "CREATE INDEX movieTmdbId IF NOT EXISTS FOR (m:Movie) ON (m.tmdbId);",
        # Add other property indexes as needed for common query patterns
        # Constraints usually cover name lookups for Person/Genre

        # Vector Index (Crucial for semantic search)
        f"""
        CREATE VECTOR INDEX moviePlotIndex IF NOT EXISTS
        FOR (m:Movie) ON (m.plotEmbedding)
        OPTIONS {{ indexConfig: {{
            `vector.dimensions`: {embedding_dim},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
    ]

# --- TMDb API Fetching Functions ---

def get_tmdb_movie_details(movie_id: int) -> Optional[Dict[str, Any]]:
    """Fetches detailed info for a single movie from TMDb, including credits."""
    if not TMDB_API_KEY:
        log.error("TMDB_API_KEY not configured.")
        return None
    try:
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY, "append_to_response": "credits"}
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        # Extract relevant details
        details = {
            "tmdbId": data.get("id"),
            "title": data.get("title"),
            "tagline": data.get("tagline"),
            "plot": data.get("overview"),
            "release_date": data.get("release_date"), # YYYY-MM-DD format
            "released": int(data["release_date"][:4]) if data.get("release_date") else None,
            "genres": [g["name"] for g in data.get("genres", [])],
            "poster_path": data.get("poster_path"),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "actors": [cast["name"] for cast in data.get("credits", {}).get("cast", [])[:10]], # Limit to top 10 actors
            "directors": [crew["name"] for crew in data.get("credits", {}).get("crew", []) if crew.get("job") == "Director"]
        }
        time.sleep(API_DELAY_SECONDS) # Respect rate limits
        return details
    except requests.exceptions.RequestException as e:
        log.error(f"TMDb API request failed for movie ID {movie_id}: {e}")
        return None
    except Exception as e:
        log.error(f"Error processing TMDb data for movie ID {movie_id}: {e}")
        return None

def get_popular_movies(max_pages: int) -> List[Dict[str, Any]]:
    """Fetches popular movie IDs from TMDb."""
    movie_ids = []
    log.info(f"Fetching popular movie IDs from TMDb (up to {max_pages} pages)...")
    for page in range(1, max_pages + 1):
        if not TMDB_API_KEY:
            log.error("TMDB_API_KEY not configured.")
            break
        try:
            url = f"{TMDB_BASE_URL}/movie/popular"
            params = {"api_key": TMDB_API_KEY, "page": page}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            page_ids = [movie["id"] for movie in data.get("results", [])]
            if not page_ids:
                log.info(f"No more movie IDs found on page {page}.")
                break
            movie_ids.extend(page_ids)
            log.info(f"Fetched page {page}/{max_pages}, total IDs so far: {len(movie_ids)}")
            time.sleep(API_DELAY_SECONDS) # Respect rate limits
        except requests.exceptions.RequestException as e:
            log.error(f"TMDb API request failed for popular movies page {page}: {e}")
            break # Stop fetching if an error occurs
        except Exception as e:
            log.error(f"Error processing TMDb popular movies page {page}: {e}")
            break
    return list(set(movie_ids)) # Return unique IDs


# --- Utility Functions ---
def get_embedding_dimension(model_name: str) -> int:
    """Gets the dimension of the configured embedding model."""
    try:
        model = get_embedding_model() 
        return model.get_sentence_embedding_dimension()
    except Exception as e:
        log.error(f"Could not determine embedding dimension for model {model_name}: {e}")
        raise ValueError(f"Failed to get embedding dimension for {model_name}") from e

# --- Neo4j Interaction Functions (Modified for clarity) ---
def run_cypher_queries(driver: Driver, queries: List[str]):
    """Executes a list of Cypher queries (for constraints/indexes)."""
    log.info(f"Executing {len(queries)} setup queries...")
    try:
        # Use bookmarks for index creation consistency if running in a cluster (Aura is clustered)
        last_bookmark = None
        with driver.session(database="neo4j") as session: # 'neo4j' database by default (free version)
            for query in queries:
                try:
                    log.info(f"Executing: {query[:100]}...") # Log start of query
                    # Use execute_write for schema changes
                    summary = session.execute_write(lambda tx: tx.run(query).consume(),
                                                  # bookmarks=(last_bookmark,) # Pass previous bookmark
                                                 )
                    log.info(f"  Query executed. Summary: {summary.counters}")
                    # last_bookmark = session.last_bookmark() # Capture bookmark
                except Exception as e:
                    # Constraints/indexes might already exist, which can cause errors
                    # Neo4jError: Unable to create CONSTRAINT... an index is already online...
                    if "already online" in str(e) or "already exists" in str(e):
                         log.warning(f"Skipping query (constraint/index likely already exists): {query[:100]}...")
                    else:
                        log.error(f"Failed to execute query: {query[:100]}... Error: {e}")
                        # Decide whether to continue or stop
                        raise e # Option: stop if any setup query fails critically
        log.info("Setup queries execution finished.")
        # return last_bookmark
    except Exception as e:
        log.error(f"An error occurred during setup query execution: {e}")
        raise # Re-raise critical error


def batch_load_movies_to_neo4j(driver: Driver, movie_data_list: List[Dict[str, Any]]):
    """Loads processed movie data into Neo4j in batches."""
    total_movies = len(movie_data_list)
    loaded_count = 0
    log.info(f"Starting batch loading of {total_movies} movies ({NEO4J_BATCH_SIZE} per batch)...")
    # last_bookmark = None # If needed for chaining writes
    try:
        with driver.session(database="neo4j") as session:
            for i in range(0, total_movies, NEO4J_BATCH_SIZE):
                batch = movie_data_list[i:min(i + NEO4J_BATCH_SIZE, total_movies)]
                log.info(f"Processing batch {i // NEO4J_BATCH_SIZE + 1}/{(total_movies + NEO4J_BATCH_SIZE - 1) // NEO4J_BATCH_SIZE}...")

                try:
                    # Use execute_write for data loading
                    summary = session.execute_write(
                        lambda tx: tx.run(LOAD_MOVIES_CYPHER, batch=batch).consume(),
                        # bookmarks=(last_bookmark,) # Pass previous bookmark if using
                    )
                    loaded_count += len(batch)
                    log.info(f"  Batch loaded. Summary: {summary.counters}. Total loaded: {loaded_count}/{total_movies}")
                    # last_bookmark = session.last_bookmark() # Capture bookmark after successful batch
                except Exception as e:
                    log.error(f"Error processing Neo4j batch starting at index {i}: {e}")
                    # Decide: Skip batch, retry, or stop?
                    log.error(f"Skipping batch due to error. Data for movies {i} to {i+NEO4J_BATCH_SIZE-1} might be missing/incomplete.")
                    # Consider adding failed batch data to a retry list

        log.info(f"Finished loading data. Successfully processed approximately {loaded_count} movies.")

    except Exception as e:
        log.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Load movie data from TMDb API into Neo4j AuraDB.")
    parser.add_argument("--max-pages", type=int, default=MAX_TMDB_PAGES, help="Maximum number of TMDb popular movie pages to fetch.")
    parser.add_argument("--batch-size", type=int, default=NEO4J_BATCH_SIZE, help="Number of movies to load into Neo4j per batch.")
    parser.add_argument("--skip-constraints", action="store_true", help="Skip creating constraints.")
    parser.add_argument("--skip-indexes", action="store_true", help="Skip creating indexes.")
    parser.add_argument("--skip-load", action="store_true", help="Skip fetching from TMDb and loading movie data.")

    args = parser.parse_args()

    # Update global constants from args if provided


    if not TMDB_API_KEY:
         log.error("TMDB_API_KEY is not set in the environment variables (.env file). Cannot fetch data.")
         return # Exit if no API key

    driver = None
    all_movie_details = []

    try:
        # --- 1. Fetch Movie IDs from TMDb ---
        if not args.skip_load:
            movie_ids = get_popular_movies(MAX_TMDB_PAGES)
            log.info(f"Found {len(movie_ids)} unique popular movie IDs to process.")

            if not movie_ids:
                log.warning("No movie IDs fetched from TMDb. Exiting data loading.")
                return

            # --- 2. Fetch Detailed Data for each Movie ID ---
            log.info("Fetching detailed movie data from TMDb...")
            processed_count = 0
            for movie_id in movie_ids:
                details = get_tmdb_movie_details(movie_id)
                if details and details.get('title') and details.get('plot'): # Basic validation
                    all_movie_details.append(details)
                    processed_count += 1
                    if processed_count % 50 == 0: # Log progress every 50 movies
                       log.info(f"Fetched details for {processed_count}/{len(movie_ids)} movies...")
                else:
                    log.warning(f"Skipping movie ID {movie_id} due to missing data or fetch error.")
            log.info(f"Successfully fetched details for {len(all_movie_details)} movies.")

            if not all_movie_details:
                log.warning("No valid movie details collected after fetching. Exiting.")
                return

            # --- 3. Generate Embeddings ---
            log.info("Generating plot embeddings...")
            plots = [movie.get('plot', '') for movie in all_movie_details]
            embeddings = [generate_embedding(plot) for plot in plots] # Consider batching 'encode' for large lists
            log.info("Embeddings generated.")

            # Add embeddings to movie data
            for i, movie in enumerate(all_movie_details):
                movie['plotEmbedding'] = embeddings[i]

        # --- 4. Connect to Neo4j ---
        driver = get_driver() # Establish connection only when needed
        log.info("Connected to Neo4j AuraDB.")

        # --- 5. Create Constraints ---
        if not args.skip_constraints:
            log.info("Attempting to create constraints...")
            run_cypher_queries(driver, CREATE_CONSTRAINTS_CYPHER)
        else:
            log.info("Skipping constraint creation.")

        # --- 6. Load Data into Neo4j ---
        if not args.skip_load and all_movie_details:
            batch_load_movies_to_neo4j(driver, all_movie_details)
        elif args.skip_load:
            log.info("Skipping data loading as requested.")
        else:
             log.warning("Skipping data loading because no movie details were fetched/processed.")


        # --- 7. Create Indexes (including Vector Index) ---
        # IMPORTANT: Create indexes *after* data is loaded so nodes with the
        # embedding property exist for the vector index.
        if not args.skip_indexes:
            log.info("Attempting to create indexes...")
            try:
                embedding_dim = get_embedding_dimension(settings.EMBEDDING_MODEL_NAME)
                index_queries = get_create_indexes_cypher(embedding_dim)
                run_cypher_queries(driver, index_queries)
            except ValueError as e:
                 log.error(f"Cannot create indexes: {e}")
            except Exception as e: # Catch errors during index creation specifically
                 log.error(f"Failed during index creation phase: {e}")
        else:
            log.info("Skipping index creation.")

        log.info("Data loading script finished.")

    except ConnectionError as ce:
        log.error(f"Failed to connect to Neo4j: {ce}")
    except ValueError as ve: # Catch specific errors like missing embedding dimension
         log.error(f"Configuration or Value error: {ve}")
    except Exception as e:
        log.error(f"An unexpected error occurred during the loading process: {e}", exc_info=True)
    finally:
        if driver:
            close_driver()
            log.info("Neo4j connection closed.")


if __name__ == "__main__":
    main()