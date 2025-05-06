# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #this file is for creating database on free auradb and feeding data into them
# #
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from config import settings
# from langchain_community.graphs import Neo4jGraph
# import requests
# import logging
# from typing import Dict, List, Optional

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TMDBMovieDatabase:
#     def __init__(self):
#         self.kg = Neo4jGraph(
#             url=settings.NEO4J_URI,
#             username=settings.NEO4J_USERNAME,
#             password=settings.NEO4J_PASSWORD,
#             database="neo4j"
#         )
#         self.tmdb_api_key = settings.TMDB_API_KEY
#         self.base_url = "https://api.themoviedb.org/3"
#         self.image_base_url = "https://image.tmdb.org/t/p/w500"

#     def create_constraints(self) -> None:
#         constraints = [
#             "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdb_id IS UNIQUE",
#             "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.tmdb_id IS UNIQUE"
#         ]
#         for constraint in constraints:
#             self.kg.query(constraint)
#         logger.info("Created database constraints")

#     def fetch_movie_data(self, movie_id: int) -> Optional[Dict]:
#         try:
#             movie_url = f"{self.base_url}/movie/{movie_id}"
#             params = {
#                 "api_key": self.tmdb_api_key,
#                 "append_to_response": "credits"
#             }
#             response = requests.get(movie_url, params=params, timeout=10)
#             response.raise_for_status()
#             data = response.json()

#             poster_path = data.get("poster_path", "")
#             poster_url = f"{self.image_base_url}{poster_path}" if poster_path else ""

#             return {
#                 "tmdb_id": data["id"],
#                 "title": data["title"],
#                 "tagline": data.get("tagline", ""),
#                 "overview": data.get("overview", ""),
#                 "release_date": data.get("release_date", ""),
#                 "poster_path": poster_path,
#                 "poster_url": poster_url,
#                 "directors": [
#                     crew for crew in data["credits"]["crew"] 
#                     if crew["job"] == "Director"
#                 ],
#                 "cast": data["credits"]["cast"][:10]
#             }
#         except Exception as e:
#             logger.error(f"Failed to fetch TMDB data for movie {movie_id}: {e}")
#             return None
#     def fetch_trailer_url(self, movie_id: int) -> Optional[str]:
#         """Fetch trailer URL (YouTube) from TMDb"""
#         try:
#             url = f"{self.base_url}/movie/{movie_id}/videos"
#             params = {
#                 "api_key": self.tmdb_api_key,
#                 "language": "en-US"
#             }
#             response = requests.get(url, params=params, timeout=10)
#             response.raise_for_status()
#             videos = response.json().get("results", [])
        
#             for video in videos:
#                 if video["site"] == "YouTube" and video["type"] == "Trailer":
#                     return f"https://www.youtube.com/watch?v={video['key']}"
        
#             return None  # No valid trailer found
#         except Exception as e:
#             logger.warning(f"No trailer found for movie {movie_id}: {e}")
#             return None
#     def fetch_video_url(self, movie_id:int)->Optional[str]:
#         """fetch url for recommendation"""
#         try: 
#             url = f"{self.base_url}/movie/{movie_id}/videos"
#             params = {
#             "api_key": self.tmdb_api_key,
#             "language": "en-US"
#             }
#             response = requests.get(url, params=params, timeout=10)
#             response.raise_for_status()
#             videos = response.json().get("results", [])
#             for video in videos:
#                 if video["site"] == "YouTube" and video["type"] == "Trailer":
#                     return f"https://www.youtube.com/watch?v={video['key']}"
#             return None  # No valid trailer found
#         except Exception as e:
#             logger.warning(f"No trailer found for movie {movie_id}: {e}")
#             return None    
            
    
#     def create_movie_from_tmdb(self, movie_id: int) -> bool:
#         movie_data = self.fetch_movie_data(movie_id)
#         if not movie_data:
#             return False
#         trailer_url = self.fetch_trailer_url(movie_id)

#         try:
#             self.kg.query("""
#             MERGE (m:Movie {tmdb_id: $tmdb_id})
#             SET m.title = $title,
#                 m.tagline = $tagline,
#                 m.overview = $overview,
#                 m.release_date = date($release_date),
#                 m.poster_path = $poster_path,
#                 m.poster_url = $poster_url,
#                 m.trailer_url = $trailer_url
#             """, params={
#                 "tmdb_id": movie_data["tmdb_id"],
#                 "title": movie_data["title"],
#                 "tagline": movie_data["tagline"],
#                 "overview": movie_data["overview"],
#                 "release_date": movie_data["release_date"],
#                 "poster_path": movie_data["poster_path"],
#                 "poster_url": movie_data["poster_url"],
#                 "trailer_url": trailer_url
#             })

#             for director in movie_data["directors"]:
#                 self._create_person_relationship(
#                     movie_data["tmdb_id"],
#                     director["id"],
#                     director["name"],
#                     director.get("gender", 0),
#                     director.get("birthday", ""),
#                     "DIRECTED",
#                     None
#                 )

#             for actor in movie_data["cast"]:
#                 self._create_person_relationship(
#                     movie_data["tmdb_id"],
#                     actor["id"],
#                     actor["name"],
#                     actor.get("gender", 0),
#                     actor.get("birthday", ""),
#                     "ACTED_IN",
#                     [actor.get("character", "")]
#                 )

#             logger.info(f"Created movie: {movie_data['title']}")
#             return True

#         except Exception as e:
#             logger.error(f"Failed to create movie {movie_id}: {e}")
#             return False

#     def _create_person_relationship(self, movie_id: int, person_id: int, name: str, 
#                                   gender: int, birthday: str, 
#                                   rel_type: str, roles: Optional[List[str]]) -> None:
#         self.kg.query("""
#         MERGE (p:Person {tmdb_id: $person_id})
#         SET p.name = $name,
#             p.gender = $gender,
#             p.birthday = date($birthday)
#         """, params={
#             "person_id": person_id,
#             "name": name,
#             "gender": gender,
#             "birthday": birthday if birthday else None
#         })

#         if rel_type == "ACTED_IN" and roles:
#             self.kg.query("""
#             MATCH (m:Movie {tmdb_id: $movie_id})
#             MATCH (p:Person {tmdb_id: $person_id})
#             MERGE (p)-[r:ACTED_IN]->(m)
#             SET r.roles = $roles
#             """, params={
#                 "movie_id": movie_id,
#                 "person_id": person_id,
#                 "roles": roles
#             })
#         else:
#             self.kg.query(f"""
#             MATCH (m:Movie {{tmdb_id: $movie_id}})
#             MATCH (p:Person {{tmdb_id: $person_id}})
#             MERGE (p)-[:{rel_type}]->(m)
#             """, params={
#                 "movie_id": movie_id,
#                 "person_id": person_id
#             })

#     def import_popular_movies(self, count: int = 5) -> None:
#         try:
#             url = f"{self.base_url}/movie/popular"
#             params = {
#                 "api_key": self.tmdb_api_key,
#                 "language": "en-US",
#                 "page": 1
#             }
#             response = requests.get(url, params=params, timeout=10)
#             response.raise_for_status()
#             movies = response.json()["results"][:count]
            
#             for movie in movies:
#                 success = self.create_movie_from_tmdb(movie["id"])
#                 logger.info(f"Movie {movie['title']} imported: {'Success' if success else 'Failed'}")
                
#         except Exception as e:
#             logger.error(f"Failed to fetch popular movies: {e}")



#     def close(self) -> None:
#         self.kg.close()

# def main():
#     db = TMDBMovieDatabase()
#     try:
#         db.create_constraints()
#         db.import_popular_movies(count=5)
#         logger.info("Database population completed")
#     except Exception as e:
#         logger.error(f"Error: {e}")

# if __name__ == '__main__':
#     main()
import time
from neo4j import GraphDatabase, Transaction
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Optional, Set
import logging
import requests
from random import uniform
from datetime import datetime
from config import settings

# from config import settings


class Settings:
    NEO4J_URI = settings.NEO4J_URI
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = settings.NEO4J_PASSWORD
    TMDB_API_KEY = settings.TMDB_API_KEY 
    LOG_LEVEL = "INFO"

settings = Settings()


class MovieDatabaseImporter:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
            max_connection_pool_size=20,
            connection_timeout=60
        )
        self.tmdb_api_key = settings.TMDB_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        self.batch_size = 50  # Optimal batch size for Neo4j
        self.request_delay = 0.2  # Seconds between API requests
        self._setup_logging() # This call is correct

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        # Prevent adding multiple handlers if _setup_logging is called multiple times (e.g. in tests)
        if not self.logger.handlers:
            handler = logging.FileHandler('movie_import.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            log_level_str = getattr(settings, "LOG_LEVEL", "INFO").upper()
            self.logger.setLevel(getattr(logging, log_level_str, logging.INFO))
    # _setup_logging method ends here.

    # fetch_movie_ids method starts here, correctly indented at the class level.
    def fetch_movie_ids(self, total_movies: int = 10000) -> List[int]:
        """Fetch movie IDs from TMDB using pagination"""
        movie_ids = set()
        page = 1
        movies_per_page = 20  # TMDB default

        self.logger.info(f"Starting to fetch up to {total_movies} movie IDs from TMDB.")
        while len(movie_ids) < total_movies:
            try:
                time.sleep(self.request_delay)
                url = f"{self.base_url}/movie/popular"
                params = {
                    "api_key": self.tmdb_api_key,
                    "page": page,
                    "language": "en-US"
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()
                if not data.get('results'):
                    self.logger.info("No more movie results from TMDB.")
                    break

                for movie in data['results']:
                    movie_ids.add(movie['id'])
                    if len(movie_ids) >= total_movies:
                        break

                if len(movie_ids) >= total_movies:
                    break

                page += 1
                total_pages = data.get('total_pages', 1)
                if page > total_pages:
                    self.logger.info("Reached the last page of TMDB results.")
                    break
                
                if page % 10 == 0: # Log progress every 10 pages
                    self.logger.info(f"Fetched {len(movie_ids)} movie IDs so far (page {page}).")

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching movie IDs on page {page}: {e}")
                time.sleep(5)  # Wait before retrying the same page or continuing
                continue # Or you might want to break or implement more sophisticated retry for this part
            except Exception as e: # Catch any other unexpected error
                self.logger.error(f"Unexpected error fetching movie IDs: {e}")
                time.sleep(5)
                continue
        
        self.logger.info(f"Finished fetching IDs. Total unique movie IDs fetched: {len(movie_ids)}")
        return list(movie_ids)[:total_movies]

    def import_movies(self, movie_ids: List[int], resume: bool = True):
        """Import movies in batches with transaction support"""
        imported_ids = set()
        if resume:
            try:
                imported_ids = self._get_imported_movie_ids()
                self.logger.info(f"Found {len(imported_ids)} already imported movie IDs.")
            except Exception as e:
                self.logger.error(f"Could not fetch imported movie IDs for resume: {e}. Starting fresh.")
                imported_ids = set() # Ensure it's a set even if fetching fails

        movie_ids_to_import = [mid for mid in movie_ids if mid not in imported_ids]
        total_movies_to_import = len(movie_ids_to_import)

        if not movie_ids_to_import:
            self.logger.info("No new movies to import.")
            return

        self.logger.info(f"Starting import of {total_movies_to_import} movies")

        for i in range(0, total_movies_to_import, self.batch_size):
            batch_movie_ids = movie_ids_to_import[i:i + self.batch_size]
            success_count = 0
            processed_in_batch = 0

            # Using a single session for the entire batch
            try:
                with self.driver.session() as session:
                    for movie_id in batch_movie_ids:
                        processed_in_batch +=1
                        try:
                            movie_data = self._fetch_movie_with_retry(movie_id)
                            if not movie_data:
                                self.logger.warning(f"Skipping movie ID {movie_id} as no data was fetched.")
                                continue

                            session.write_transaction(
                                self._create_movie_transaction,
                                movie_data
                            )
                            success_count += 1
                            self.logger.debug(f"Successfully imported movie ID {movie_id}: {movie_data.get('title')}")

                        except Exception as e:
                            # Log specific movie failure but continue with the batch
                            self.logger.error(f"Failed to process movie ID {movie_id} in batch: {str(e)}", exc_info=False) # exc_info=False to avoid too verbose logs for individual movie failures
                            continue
                
                self.logger.info(
                    f"Batch {i//self.batch_size + 1} of { (total_movies_to_import + self.batch_size -1) // self.batch_size }: "
                    f"Attempted to process {processed_in_batch} movies, "
                    f"{success_count} successful."
                )
            except Exception as e:
                self.logger.error(f"Critical error during batch {i//self.batch_size + 1} processing: {e}", exc_info=True)
                # Decide if you want to stop or continue with next batch
                # For now, it continues with the next batch.

            # Be kind to the API (if fetching happens inside the loop, though here it's pre-fetched)
            # and allow Neo4j some breathing room if writes are heavy.
            if i + self.batch_size < total_movies_to_import: # Avoid sleep after the last batch
                time.sleep(1) # Sleep between batches

    @staticmethod
    def _create_movie_transaction(tx: Transaction, movie_data: Dict):
        """Handle all database writes for a single movie"""
        # Ensure release_date and birthday are None if empty string, to avoid Cypher date conversion errors
        release_date = movie_data.get("release_date")
        if release_date == "":
            release_date = None

        # Create or update movie node
        tx.run("""
        MERGE (m:Movie {tmdb_id: $tmdb_id})
        SET m.title = $title,
            m.tagline = $tagline,
            m.overview = $overview,
            m.release_date = CASE WHEN $release_date IS NOT NULL AND $release_date <> "" THEN date($release_date) ELSE null END,
            m.poster_path = $poster_path,
            m.poster_url = $poster_url,
            m.trailer_url = $trailer_url,
            m.popularity = $popularity,
            m.vote_average = $vote_average,
            m.vote_count = $vote_count,
            m.last_updated = datetime()
        """, tmdb_id=movie_data["tmdb_id"], title=movie_data["title"], tagline=movie_data["tagline"],
               overview=movie_data["overview"], release_date=release_date, poster_path=movie_data["poster_path"],
               poster_url=movie_data["poster_url"], trailer_url=movie_data["trailer_url"],
               popularity=movie_data["popularity"], vote_average=movie_data["vote_average"],
               vote_count=movie_data["vote_count"])

        # Process directors
        if movie_data.get('directors'):
            directors_data = []
            for director in movie_data['directors']:
                birthday = director.get("birthday")
                if birthday == "":
                    birthday = None
                directors_data.append({
                    "id": director["id"],
                    "name": director["name"],
                    "gender": director.get("gender", 0), # TMDB: 0 for Not specified, 1 for Female, 2 for Male, 3 for Non-binary
                    "birthday": birthday
                })

            tx.run("""
            UNWIND $directors AS director_data
            MERGE (p:Person {tmdb_id: director_data.id})
            ON CREATE SET p.name = director_data.name,
                          p.gender = director_data.gender,
                          p.birthday = CASE WHEN director_data.birthday IS NOT NULL AND director_data.birthday <> "" THEN date(director_data.birthday) ELSE null END,
                          p.last_updated = datetime()
            ON MATCH SET  p.name = director_data.name, 
                          p.gender = director_data.gender,
                          p.birthday = CASE WHEN director_data.birthday IS NOT NULL AND director_data.birthday <> "" THEN date(director_data.birthday) ELSE null END,
                          p.last_updated = datetime()
            WITH p, $tmdb_id AS movieId
            MATCH (m:Movie {tmdb_id: movieId})
            MERGE (p)-[:DIRECTED]->(m)
            """, directors=directors_data, tmdb_id=movie_data['tmdb_id'])

        # Process cast
        if movie_data.get('cast'):
            cast_data = []
            for actor in movie_data['cast']:
                birthday = actor.get("birthday")
                if birthday == "":
                    birthday = None
                cast_data.append({
                    "id": actor["id"],
                    "name": actor["name"],
                    "gender": actor.get("gender", 0),
                    "birthday": birthday,
                    "character": actor.get("character", "")
                })

            tx.run("""
            UNWIND $cast AS actor_data
            MERGE (p:Person {tmdb_id: actor_data.id})
            ON CREATE SET p.name = actor_data.name,
                          p.gender = actor_data.gender,
                          p.birthday = CASE WHEN actor_data.birthday IS NOT NULL AND actor_data.birthday <> "" THEN date(actor_data.birthday) ELSE null END,
                          p.last_updated = datetime()
            ON MATCH SET  p.name = actor_data.name,
                          p.gender = actor_data.gender,
                          p.birthday = CASE WHEN actor_data.birthday IS NOT NULL AND actor_data.birthday <> "" THEN date(actor_data.birthday) ELSE null END,
                          p.last_updated = datetime()
            WITH p, $tmdb_id AS movieId, actor_data.character AS character_name
            MATCH (m:Movie {tmdb_id: movieId})
            MERGE (p)-[r:ACTED_IN]->(m)
            SET r.roles = CASE WHEN character_name IS NOT NULL AND character_name <> "" THEN [character_name] ELSE [] END
            """, cast=cast_data, tmdb_id=movie_data['tmdb_id'])

    @retry(
        stop=stop_after_attempt(5), # Increased attempts
        wait=wait_exponential(multiplier=1, min=2, max=30), # Increased max wait
        reraise=True)
    def _fetch_movie_with_retry(self, movie_id: int) -> Optional[Dict]:
        """Fetch movie data with retry logic and rate limiting"""
        # Random delay moved to be more effective if multiple instances/threads run
        time.sleep(uniform(self.request_delay * 0.8, self.request_delay * 1.2))

        self.logger.debug(f"Fetching movie details for ID: {movie_id}")
        try:
            movie_url = f"{self.base_url}/movie/{movie_id}"
            # Also fetching person details for directors and cast to get birthday
            # This requires multiple API calls or careful selection from 'credits' if available
            # The current 'credits' append_to_response might not have birthday for all crew/cast.
            # For simplicity, assuming 'credits' is sufficient. If not, individual person lookups would be needed.
            params = {
                "api_key": self.tmdb_api_key,
                "language": "en-US", # Added language
                "append_to_response": "credits,videos"
            }
            response = requests.get(movie_url, params=params, timeout=15)

            if response.status_code == 404:
                self.logger.warning(f"Movie ID {movie_id} not found (404).")
                return None
            if response.status_code == 401: # Unauthorized - API key issue
                self.logger.error(f"TMDB API Key Unauthorized (401) for movie ID {movie_id}. Check your API key.")
                # Potentially raise a specific exception or stop the import
                raise requests.exceptions.HTTPError("TMDB API Key Unauthorized") 
            
            response.raise_for_status() # Will raise for 4xx/5xx excluding 404 handled above
            data = response.json()

            trailer_url = None
            if data.get('videos') and data['videos'].get('results'):
                for video in data['videos']['results']:
                    if video.get('site') == 'YouTube' and video.get('type') == 'Trailer':
                        trailer_url = f"https://www.youtube.com/watch?v={video['key']}"
                        break
            
            # Fetching person details for directors and cast to get birthdays
            # This is a simplified approach. TMDB's /movie/{movie_id}/credits doesn't include birthdays.
            # A more robust solution would be to fetch each person individually if birthday is crucial
            # or accept that birthdays might be missing. For now, we'll assume it's not present in credits
            # and rely on what _create_movie_transaction can handle (nulls).
            # The original code was trying to get 'birthday' from crew/actor objects directly from movie credits.
            # This is often NOT available there.
            # For now, I'll keep the structure but acknowledge birthdays from credits are unlikely.

            directors = []
            if data.get("credits") and data["credits"].get("crew"):
                for crew_member in data["credits"]["crew"]:
                    if crew_member.get("job") == "Director":
                        directors.append({
                            "id": crew_member["id"],
                            "name": crew_member["name"],
                            "gender": crew_member.get("gender", 0),
                             # Birthday is typically NOT in movie credits.
                             # Sending empty string to be handled by _create_movie_transaction
                            "birthday": "" # Placeholder
                        })

            cast = []
            if data.get("credits") and data["credits"].get("cast"):
                for actor in data["credits"]["cast"][:15]: # Top 15 actors
                    cast.append({
                        "id": actor["id"],
                        "name": actor["name"],
                        "gender": actor.get("gender", 0),
                        "character": actor.get("character", ""),
                        "birthday": "" # Placeholder
                    })
            
            poster_path_val = data.get("poster_path")
            return {
                "tmdb_id": data["id"],
                "title": data.get("title", "N/A"),
                "tagline": data.get("tagline", ""),
                "overview": data.get("overview", ""),
                "release_date": data.get("release_date", ""), # Will be converted to date or null in Cypher
                "poster_path": poster_path_val if poster_path_val else "",
                "poster_url": f"{self.image_base_url}{poster_path_val}" if poster_path_val else "",
                "trailer_url": trailer_url,
                "popularity": data.get("popularity", 0.0),
                "vote_average": data.get("vote_average", 0.0),
                "vote_count": data.get("vote_count", 0),
                "directors": directors,
                "cast": cast
            }

        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout fetching movie ID {movie_id}. Retrying...")
            raise # Reraise to trigger tenacity
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429: # Too Many Requests
                 self.logger.warning(f"Rate limit hit (429) fetching movie {movie_id}. Retrying with longer delay...")
                 # Tenacity will handle the wait, but you could also add a specific long sleep here if needed.
                 # For TMDB, tenacity's exponential backoff should be respected.
            elif e.response.status_code != 404: # Don't log 404 as an error again if it slips through
                self.logger.error(f"HTTP error fetching movie {movie_id}: {e.response.status_code} - {e.response.text}")
            raise # Reraise to trigger tenacity or be caught by outer try-except
        except Exception as e:
            self.logger.error(f"Unexpected error fetching movie ID {movie_id}: {str(e)}", exc_info=True)
            raise # Reraise to trigger tenacity or be caught

    def _get_imported_movie_ids(self) -> Set[int]:
        """Get set of already imported movie IDs"""
        self.logger.info("Fetching already imported movie IDs from Neo4j...")
        with self.driver.session() as session:
            result = session.run("MATCH (m:Movie) WHERE m.tmdb_id IS NOT NULL RETURN m.tmdb_id as tmdb_id")
            return {record["tmdb_id"] for record in result if record["tmdb_id"] is not None}

    def create_indexes_and_constraints(self):
        """Create necessary indexes and constraints for performance and data integrity"""
        self.logger.info("Creating database indexes and constraints...")
        queries = [
            # Constraints (imply unique indexes and enforce uniqueness)
            "CREATE CONSTRAINT movie_tmdb_id_unique IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdb_id IS UNIQUE",
            "CREATE CONSTRAINT person_tmdb_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.tmdb_id IS UNIQUE",
            
            # Additional indexes for querying
            "CREATE INDEX movie_title IF NOT EXISTS FOR (m:Movie) ON (m.title)",
            "CREATE INDEX movie_release_date IF NOT EXISTS FOR (m:Movie) ON (m.release_date)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)"
        ]
        try:
            with self.driver.session() as session:
                for query in queries:
                    self.logger.debug(f"Running: {query}")
                    session.run(query)
            self.logger.info("Successfully created/verified database indexes and constraints.")
        except Exception as e:
            self.logger.error(f"Error creating indexes/constraints: {e}", exc_info=True)
            # Depending on the error, you might want to raise it or handle it
            raise

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            self.logger.info("Neo4j driver connection closed.")

def main():
    importer = None # Ensure importer is defined for finally block
    try:
        importer = MovieDatabaseImporter()
        importer.logger.info("MovieDatabaseImporter initialized.")
        
        # Step 1: Setup database (constraints are crucial before data import)
        importer.create_indexes_and_constraints()
        
        # Step 2: Fetch movie IDs
        # For testing, fetch a smaller number:
        # movie_ids = importer.fetch_movie_ids(total_movies=100)
        movie_ids = importer.fetch_movie_ids(total_movies=5000) # For full run
        
        if not movie_ids:
            importer.logger.info("No movie IDs fetched. Exiting.")
            return

        # Step 3: Import movies with resume capability
        importer.import_movies(movie_ids, resume=True)
        
        importer.logger.info("Movie import process completed.")
        
    except Exception as e:
        if importer and hasattr(importer, 'logger'):
            importer.logger.error(f"A fatal error occurred in the main process: {str(e)}", exc_info=True)
        else:
            # Fallback logger if importer or its logger is not initialized
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"A fatal error occurred before logger fully initialized: {str(e)}", exc_info=True)
    finally:
        if importer:
            importer.close()

if __name__ == '__main__':
    # Basic logging setup for messages before the class logger is ready or if it fails
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()