import argparse
import logging
import time
import requests
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Tuple
from neo4j import Driver, Transaction

# --- Project Imports ---
try:
    from graph_db.connection import get_driver, close_driver
    from embeddings.generator import (
        generate_text_embedding,
        generate_image_embedding_from_url, # Used for POSTERS and now THUMBNAILS
        get_text_embedding_model
    )
    from multimodal.vision_model import _get_vision_model_and_processor # For image dimension
    from config import settings
except ImportError as e:
    print(f"Import Error: {e}. Make sure you are running from the project root using 'python -m scripts.load_data' and all __init__.py files exist.")
    exit()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_NEO4J_BATCH_SIZE = 1
DEFAULT_MAX_TMDB_PAGES = 1
API_DELAY_SECONDS = 0.35 # Keep slightly higher delay

# --- TMDb API Configuration ---
TMDB_API_KEY = settings.TMDB_API_KEY
TMDB_BASE_URL = "https://api.themoviedb.org/3"
# YouTube Thumbnail URL Pattern (mqdefault is medium quality)
YT_THUMB_URL_PATTERN = "https://img.youtube.com/vi/{key}/mqdefault.jpg"

# --- Cypher Queries ---
CREATE_CONSTRAINTS_CYPHER = [
    "CREATE CONSTRAINT movieTitle IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE;",
    "CREATE CONSTRAINT movieTmdbId IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdbId IS UNIQUE;",
    "CREATE CONSTRAINT personName IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE;",
    "CREATE CONSTRAINT genreName IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;",
]

# Updated Cypher to include trailerKey AND trailerThumbnailEmbedding
LOAD_MOVIES_CYPHER = """
UNWIND $batch AS movie_data

MERGE (m:Movie {tmdbId: movie_data.tmdbId})
ON CREATE SET
    m.title = movie_data.title,
    m.released = movie_data.released,
    m.release_date = movie_data.release_date,
    m.tagline = movie_data.tagline,
    m.plot = movie_data.plot,
    m.poster_path = movie_data.poster_path,
    m.trailerKey = movie_data.trailerKey,                 // Trailer key
    m.vote_average = movie_data.vote_average,
    m.vote_count = movie_data.vote_count,
    m.plotEmbedding = movie_data.plotEmbedding,           // Text embedding
    m.posterEmbedding = movie_data.posterEmbedding,       // Poster image embedding
    m.trailerThumbnailEmbedding = movie_data.trailerThumbnailEmbedding // <--- NEW: Thumbnail embedding
ON MATCH SET
    m.title = movie_data.title,
    m.released = movie_data.released,
    m.release_date = movie_data.release_date,
    m.tagline = movie_data.tagline,
    m.plot = movie_data.plot,
    m.poster_path = movie_data.poster_path,
    m.trailerKey = movie_data.trailerKey,                 // Update trailer key
    m.vote_average = movie_data.vote_average,
    m.vote_count = movie_data.vote_count,
    m.plotEmbedding = movie_data.plotEmbedding,           // Update embedding too
    m.posterEmbedding = movie_data.posterEmbedding,       // Update embedding too
    m.trailerThumbnailEmbedding = movie_data.trailerThumbnailEmbedding // <--- NEW: Update thumbnail embedding

// Merge Director, Genre, Actor nodes/relationships (Keep as before)
FOREACH (director_name IN movie_data.directors WHERE director_name IS NOT NULL AND director_name <> '' | MERGE (d:Person {name: director_name}) MERGE (d)-[:DIRECTED]->(m))
FOREACH (genre_name IN movie_data.genres WHERE genre_name IS NOT NULL AND genre_name <> '' | MERGE (g:Genre {name: genre_name}) MERGE (m)-[:HAS_GENRE]->(g))
FOREACH (actor_info IN movie_data.actors_data WHERE actor_info IS NOT NULL AND actor_info.name IS NOT NULL AND actor_info.name <> '' |
    MERGE (a:Person {name: actor_info.name})
    ON CREATE SET a.dob = CASE WHEN actor_info.dob IS NOT NULL THEN actor_info.dob ELSE null END
    ON MATCH SET a.dob = CASE WHEN a.dob IS NULL AND actor_info.dob IS NOT NULL THEN actor_info.dob ELSE a.dob END
    MERGE (a)-[:ACTED_IN]->(m)
)
"""

# Function to dynamically generate index queries including ALL THREE vector indexes
def get_create_indexes_cypher(text_embedding_dim: int, image_embedding_dim: int) -> List[str]:
    # Assuming trailer thumbnail uses the same CLIP model dimension as posters
    trailer_thumb_embedding_dim = image_embedding_dim
    return [
        "CREATE INDEX movieReleased IF NOT EXISTS FOR (m:Movie) ON (m.released);",
        "CREATE INDEX movieTmdbId IF NOT EXISTS FOR (m:Movie) ON (m.tmdbId);",
        "CREATE INDEX personDob IF NOT EXISTS FOR (p:Person) ON (p.dob);",

        # Text Vector Index
        f"""CREATE VECTOR INDEX moviePlotIndex IF NOT EXISTS FOR (m:Movie) ON (m.plotEmbedding)
           OPTIONS {{ indexConfig: {{ `vector.dimensions`: {text_embedding_dim}, `vector.similarity_function`: 'cosine' }} }}""",
        # Poster Image Vector Index
        f"""CREATE VECTOR INDEX moviePosterIndex IF NOT EXISTS FOR (m:Movie) ON (m.posterEmbedding)
           OPTIONS {{ indexConfig: {{ `vector.dimensions`: {image_embedding_dim}, `vector.similarity_function`: 'cosine' }} }}""",
        # Trailer Thumbnail Image Vector Index  <--- NEW INDEX
        f"""CREATE VECTOR INDEX movieTrailerThumbIndex IF NOT EXISTS FOR (m:Movie) ON (m.trailerThumbnailEmbedding)
           OPTIONS {{ indexConfig: {{ `vector.dimensions`: {trailer_thumb_embedding_dim}, `vector.similarity_function`: 'cosine' }} }}""",
    ]

# --- TMDb API Fetching Functions ---
def get_tmdb_movie_details(movie_id: int) -> Optional[Dict[str, Any]]:
    if not TMDB_API_KEY: log.error("TMDB_API_KEY not configured."); return None
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"; params = {"api_key": TMDB_API_KEY, "append_to_response": "credits,videos"}
    log.debug(f"Requesting details, credits, videos for TMDb ID: {movie_id}")
    try:
        response = requests.get(url, params=params, timeout=15); response.raise_for_status(); data = response.json()
        release_date = data.get("release_date"); release_year = None
        if release_date and isinstance(release_date, str) and len(release_date) >= 4:
            try: release_year = int(release_date[:4])
            except ValueError: log.warning(f"Could not parse year from release_date '{release_date}' for movie ID {movie_id}")
        trailer_key = None; videos = data.get("videos", {}).get("results", [])
        for video in videos:
            if video and video.get("site") == "YouTube" and video.get("type") == "Trailer" and video.get("official") is True: trailer_key = video.get("key"); break
        if not trailer_key:
             for video in videos:
                 if video and video.get("site") == "YouTube" and video.get("type") == "Trailer": trailer_key = video.get("key"); log.debug(f"Using first non-official YouTube trailer key for {movie_id}"); break
        if not trailer_key:
             for video in videos:
                 if video and video.get("site") == "YouTube": trailer_key = video.get("key"); log.debug(f"Using first YouTube video key (non-trailer) for {movie_id}"); break
        actors_raw = data.get("credits", {}).get("cast", [])[:15]; actors_list = []
        if actors_raw: actors_list = [{"name": cast["name"], "id": cast["id"]} for cast in actors_raw if cast and cast.get("name") and cast.get("id") and cast.get("known_for_department") == "Acting"]
        details = { "tmdbId": data.get("id"), "title": data.get("title"), "tagline": data.get("tagline"), "plot": data.get("overview"), "release_date": release_date, "released": release_year, "genres": [g["name"] for g in data.get("genres", []) if g and g.get("name")], "poster_path": data.get("poster_path"), "vote_average": data.get("vote_average"), "vote_count": data.get("vote_count"), "actors": actors_list, "directors": [crew["name"] for crew in data.get("credits", {}).get("crew", []) if crew and crew.get("job") == "Director" and crew.get("name")], "trailerKey": trailer_key }
        if not details["tmdbId"] or not details["title"]: log.warning(f"Missing essential data (ID or Title) for movie fetch with original ID {movie_id}. Skipping."); return None
        time.sleep(API_DELAY_SECONDS); return details
    except requests.exceptions.Timeout: log.error(f"TMDb API request timed out for movie ID {movie_id}"); return None
    except requests.exceptions.RequestException as e: log.error(f"TMDb API request failed for movie ID {movie_id}: {e}"); return None
    except Exception as e: log.error(f"Error processing TMDb data for movie ID {movie_id}: {e}", exc_info=True); return None

def get_person_details(person_id: int) -> Optional[Dict[str, Any]]:
    if not TMDB_API_KEY: log.error("TMDB_API_KEY not configured."); return None
    url = f"{TMDB_BASE_URL}/person/{person_id}"; params = {"api_key": TMDB_API_KEY}
    log.debug(f"Requesting details for Person ID: {person_id}")
    try:
        response = requests.get(url, params=params, timeout=10); response.raise_for_status(); data = response.json()
        person_details = { "id": data.get("id"), "name": data.get("name"), "dob": data.get("birthday") }
        if person_details["dob"] and (not isinstance(person_details["dob"], str) or len(person_details["dob"]) != 10): log.warning(f"Invalid DOB format '{person_details['dob']}' received for person {person_id}. Setting to null."); person_details["dob"] = None
        time.sleep(API_DELAY_SECONDS); return person_details
    except requests.exceptions.Timeout: log.error(f"TMDb API request timed out for person ID {person_id}"); return None
    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 404: log.warning(f"Person ID {person_id} not found on TMDb.")
        else: log.error(f"TMDb API request failed for person ID {person_id}: {e}")
        return None
    except Exception as e: log.error(f"Error processing TMDb person data for ID {person_id}: {e}", exc_info=True); return None

def get_popular_movies(max_pages_to_fetch: int) -> List[int]:
    # (Implementation from previous version)
    movie_ids = []; log.info(f"Fetching popular movie IDs from TMDb (up to {max_pages_to_fetch} pages)...")
    for page in range(1, max_pages_to_fetch + 1):
        if not TMDB_API_KEY: log.error("TMDB_API_KEY not configured."); break
        url = f"{TMDB_BASE_URL}/movie/popular"; params = {"api_key": TMDB_API_KEY, "page": page}
        try:
            response = requests.get(url, params=params, timeout=10); response.raise_for_status(); data = response.json()
            page_ids = [movie["id"] for movie in data.get("results", []) if movie and movie.get("id")]
            if not page_ids: log.info(f"No more movie IDs found on page {page}."); break
            movie_ids.extend(page_ids); log.info(f"Fetched page {page}/{max_pages_to_fetch}, total unique IDs so far: {len(set(movie_ids))}"); time.sleep(API_DELAY_SECONDS)
        except requests.exceptions.Timeout: log.error(f"TMDb API request timed out for popular movies page {page}"); break
        except requests.exceptions.RequestException as e: log.error(f"TMDb API request failed for popular movies page {page}: {e}"); break
        except Exception as e: log.error(f"Error processing TMDb popular movies page {page}: {e}", exc_info=True); break
    unique_ids = list(set(movie_ids)); log.info(f"Finished fetching popular IDs. Total unique IDs collected: {len(unique_ids)}"); return unique_ids

# (run_cypher_queries, batch_load_movies_to_neo4j remain the same)
def run_cypher_queries(driver: Driver, queries: List[str]):
    log.info(f"Executing {len(queries)} setup queries..."); 
    try:
        with driver.session(database="neo4j") as session:
            for query in queries:
                try: log.info(f"Executing: {query[:100]}..."); summary = session.execute_write(lambda tx: tx.run(query).consume()); log.info(f"  Query executed. Counters: {summary.counters}")
                except Exception as e:
                    if "already online" in str(e) or "already exists" in str(e): log.warning(f"Skipping query (constraint/index likely already exists): {query[:100]}...")
                    else: log.error(f"Failed to execute query: {query[:100]}... Error: {e}"); raise e
        log.info("Setup queries execution finished.")
    except Exception as e: log.error(f"An error occurred during setup query execution: {e}", exc_info=True); raise

def batch_load_movies_to_neo4j(driver: Driver, movie_data_list: List[Dict[str, Any]], batch_size: int):
    total_movies = len(movie_data_list); loaded_count = 0; log.info(f"Starting batch loading of {total_movies} movies ({batch_size} per batch)..."); 
    try:
        with driver.session(database="neo4j") as session:
            for i in range(0, total_movies, batch_size):
                batch = movie_data_list[i:min(i + batch_size, total_movies)]; log.info(f"Processing batch {i // batch_size + 1}/{(total_movies + batch_size - 1) // batch_size}...")
                try: summary = session.execute_write(lambda tx: tx.run(LOAD_MOVIES_CYPHER, batch=batch).consume()); loaded_count += len(batch); log.info(f"  Batch loaded. Counters: {summary.counters}. Total loaded: {loaded_count}/{total_movies}")
                except Exception as e: log.error(f"Error processing Neo4j batch starting at index {i}: {e}", exc_info=True); log.error(f"Skipping batch due to error.")
        log.info(f"Finished loading data. Successfully processed approximately {loaded_count} movies.")
    except Exception as e: log.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)


# --- Main Execution Logic ---
def main(args):
    """Main function to orchestrate data fetching, processing, and loading."""
    if not TMDB_API_KEY: log.error("TMDB_API_KEY is not set."); return

    driver = None
    all_movie_data_raw: List[Dict[str, Any]] = []
    unique_actors: Set[Tuple[int, str]] = set()

    try:
        # === Stage 1: Fetch Movie Data & Collect Unique Actors ===
        if not args.skip_load:
            # Fetch movie data (including trailer keys and actor IDs)
            # (Code remains the same as previous version)
            movie_ids = get_popular_movies(args.max_pages)
            if not movie_ids: log.warning("No movie IDs fetched."); return

            log.info(f"Fetching details for {len(movie_ids)} movies...")
            fetch_count = 0
            for movie_id in movie_ids:
                details = get_tmdb_movie_details(movie_id)
                if details:
                    all_movie_data_raw.append(details)
                    for actor in details.get('actors', []):
                        if actor and actor.get('id') and actor.get('name'): unique_actors.add((actor['id'], actor['name']))
                    fetch_count += 1
                    if fetch_count % 50 == 0: log.info(f"Fetched details for {fetch_count}/{len(movie_ids)} movies...")
                else: log.warning(f"Failed to fetch details for movie ID {movie_id}.")
            log.info(f"Finished fetching movie details. Got data for {len(all_movie_data_raw)} movies.")
            log.info(f"Found {len(unique_actors)} unique actors to fetch DOB for.")
            if not all_movie_data_raw: log.warning("No movie data collected."); return

        # === Stage 2: Fetch Actor DOBs (if loading data) ===
        actor_dob_map: Dict[int, Optional[str]] = {}
        if not args.skip_load and unique_actors:
            # Fetch DOBs for unique actors
            # (Code remains the same as previous version)
            log.info("Fetching Date of Birth for unique actors...")
            dob_fetch_count = 0
            for actor_id, actor_name in unique_actors:
                person_details = get_person_details(actor_id)
                if person_details: actor_dob_map[actor_id] = person_details.get('dob')
                dob_fetch_count +=1
                if dob_fetch_count % 50 == 0: log.info(f"Fetched DOB info for {dob_fetch_count}/{len(unique_actors)} actors...")
            log.info("Finished fetching actor DOBs.")

        # === Stage 3: Prepare Final Data for Batching ===
        final_batch_data: List[Dict[str, Any]] = []
        if not args.skip_load:
            log.info("Preparing final data list with embeddings (Text, Poster, Trailer Thumbnail) and actor DOBs...")
            prep_count = 0
            skip_prep_count = 0
            for movie_details in all_movie_data_raw:
                 plot_embedding = generate_text_embedding(movie_details.get('plot', ''))
                 if not plot_embedding:
                     log.warning(f"Skipping movie '{movie_details.get('title')}' due to failed plot embedding generation.")
                     skip_prep_count += 1
                     continue

                 #poster_embedding = generate_image_embedding_from_url(movie_details.get('poster_path'))

                 # --- NEW: Generate Trailer Thumbnail Embedding ---
                #  trailer_thumb_embedding = [] # Default to empty list
                #  trailer_key = movie_details.get('trailerKey')
                #  if trailer_key:
                #      thumbnail_url = YT_THUMB_URL_PATTERN.format(key=trailer_key)
                #      log.debug(f"Attempting to generate embedding for trailer thumbnail: {thumbnail_url}")
                #      # Reuse the image embedding function
                #      trailer_thumb_embedding = generate_image_embedding_from_url(thumbnail_url) # Pass URL directly
                #      if not trailer_thumb_embedding:
                #           log.warning(f"Failed to generate thumbnail embedding for trailer key {trailer_key}")
                 # -----------------------------------------------

                 actors_data_for_cypher = []
                 for actor in movie_details.get('actors', []):
                     actor_id = actor.get('id')
                     actor_name = actor.get('name')
                     if actor_id and actor_name:
                         actor_dob = actor_dob_map.get(actor_id)
                         actors_data_for_cypher.append({"name": actor_name, "dob": actor_dob})

                 final_movie_data = {
                     "tmdbId": movie_details["tmdbId"],
                     "title": movie_details["title"],
                     "tagline": movie_details.get("tagline"),
                     "plot": movie_details.get("plot"),
                     "release_date": movie_details.get("release_date"),
                     "released": movie_details.get("released"),
                     "genres": movie_details.get("genres", []),
                     "poster_path": movie_details.get("poster_path"),
                     "vote_average": movie_details.get("vote_average"),
                     "vote_count": movie_details.get("vote_count"),
                     "directors": movie_details.get("directors", []),
                     "actors_data": actors_data_for_cypher,
                     "plotEmbedding": plot_embedding,
                     "posterEmbedding": poster_embedding if poster_embedding else []
                     
                 }
                 final_batch_data.append(final_movie_data)
                 prep_count += 1
                 if prep_count % 50 == 0: # Log more frequently during embedding generation
                     log.info(f"Prepared final data for {prep_count}/{len(all_movie_data_raw)} movies...")
            log.info(f"Finished preparing final data. {len(final_batch_data)} movies ready for loading (Skipped: {skip_prep_count}).")
            if not final_batch_data: log.warning("No final data prepared. Exiting."); return

        # === Stage 4, 5, 6, 7 (Connect, Constraints, Load, Indexes) ===
        # (Remain the same as previous version, but ensure indexes are created correctly)
        driver = get_driver(); log.info("Connected to Neo4j.")
        if not args.skip_constraints: run_cypher_queries(driver, CREATE_CONSTRAINTS_CYPHER)
        else: log.info("Skipping constraint creation.")
        if not args.skip_load and final_batch_data: batch_load_movies_to_neo4j(driver, final_batch_data, args.batch_size)
        elif args.skip_load: log.info("Skipping data loading.")
        else: log.warning("Skipping data loading - no data prepared.")
        if not args.skip_indexes:
            log.info("Attempting to create indexes (Text, Poster, Trailer Thumb Vector, DOB)...")
            try:
                run_cypher_queries(driver, index_queries)
            except ValueError as e: log.error(f"Cannot create indexes - failed to get embedding dimensions: {e}")
            except Exception as e: log.error(f"Failed during index creation phase: {e}", exc_info=True)
        else: log.info("Skipping index creation.")

        log.info("Data loading script finished successfully.")

    except ConnectionError as ce: log.error(f"Neo4j Connection Error: {ce}")
    except ValueError as ve: log.error(f"Configuration/Value Error: {ve}")
    except Exception as e: log.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if driver: close_driver(); log.info("Neo4j connection closed.")

# --- Argument Parsing and Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load movie data (inc. embeddings[plot,poster,trailerThumb], actor DOBs, trailer keys) from TMDb API into Neo4j.")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_TMDB_PAGES, help=f"Max TMDb popular movie pages (default: {DEFAULT_MAX_TMDB_PAGES}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_NEO4J_BATCH_SIZE, help=f"Neo4j load batch size (default: {DEFAULT_NEO4J_BATCH_SIZE}).")
    parser.add_argument("--skip-constraints", action="store_true", help="Skip creating constraints.")
    parser.add_argument("--skip-indexes", action="store_true", help="Skip creating indexes.")
    parser.add_argument("--skip-load", action="store_true", help="Skip fetching/processing/loading data.")
    args = parser.parse_args()
    main(args)