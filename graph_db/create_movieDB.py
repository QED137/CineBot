#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#this file is for creating database on free auradb and feeding data into them
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from config import settings
from langchain_community.graphs import Neo4jGraph
import requests
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TMDBMovieDatabase:
    def __init__(self):
        self.kg = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database="neo4j"
        )
        self.tmdb_api_key = settings.TMDB_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"

    def create_constraints(self) -> None:
        constraints = [
            "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdb_id IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.tmdb_id IS UNIQUE"
        ]
        for constraint in constraints:
            self.kg.query(constraint)
        logger.info("Created database constraints")

    def fetch_movie_data(self, movie_id: int) -> Optional[Dict]:
        try:
            movie_url = f"{self.base_url}/movie/{movie_id}"
            params = {
                "api_key": self.tmdb_api_key,
                "append_to_response": "credits"
            }
            response = requests.get(movie_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            poster_path = data.get("poster_path", "")
            poster_url = f"{self.image_base_url}{poster_path}" if poster_path else ""

            return {
                "tmdb_id": data["id"],
                "title": data["title"],
                "tagline": data.get("tagline", ""),
                "overview": data.get("overview", ""),
                "release_date": data.get("release_date", ""),
                "poster_path": poster_path,
                "poster_url": poster_url,
                "directors": [
                    crew for crew in data["credits"]["crew"] 
                    if crew["job"] == "Director"
                ],
                "cast": data["credits"]["cast"][:10]
            }
        except Exception as e:
            logger.error(f"Failed to fetch TMDB data for movie {movie_id}: {e}")
            return None
    def fetch_trailer_url(self, movie_id: int) -> Optional[str]:
        """Fetch trailer URL (YouTube) from TMDb"""
        try:
            url = f"{self.base_url}/movie/{movie_id}/videos"
            params = {
                "api_key": self.tmdb_api_key,
                "language": "en-US"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            videos = response.json().get("results", [])
        
            for video in videos:
                if video["site"] == "YouTube" and video["type"] == "Trailer":
                    return f"https://www.youtube.com/watch?v={video['key']}"
        
            return None  # No valid trailer found
        except Exception as e:
            logger.warning(f"No trailer found for movie {movie_id}: {e}")
            return None
    def fetch_video_url(self, movie_id:int)->Optional[str]:
        """fetch url for recommendation"""
        try: 
            url = f"{self.base_url}/movie/{movie_id}/videos"
            params = {
            "api_key": self.tmdb_api_key,
            "language": "en-US"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            videos = response.json().get("results", [])
            for video in videos:
                if video["site"] == "YouTube" and video["type"] == "Trailer":
                    return f"https://www.youtube.com/watch?v={video['key']}"
            return None  # No valid trailer found
        except Exception as e:
            logger.warning(f"No trailer found for movie {movie_id}: {e}")
            return None    
            
    
    def create_movie_from_tmdb(self, movie_id: int) -> bool:
        movie_data = self.fetch_movie_data(movie_id)
        if not movie_data:
            return False
        trailer_url = self.fetch_trailer_url(movie_id)

        try:
            self.kg.query("""
            MERGE (m:Movie {tmdb_id: $tmdb_id})
            SET m.title = $title,
                m.tagline = $tagline,
                m.overview = $overview,
                m.release_date = date($release_date),
                m.poster_path = $poster_path,
                m.poster_url = $poster_url,
                m.trailer_url = $trailer_url
            """, params={
                "tmdb_id": movie_data["tmdb_id"],
                "title": movie_data["title"],
                "tagline": movie_data["tagline"],
                "overview": movie_data["overview"],
                "release_date": movie_data["release_date"],
                "poster_path": movie_data["poster_path"],
                "poster_url": movie_data["poster_url"],
                "trailer_url": trailer_url
            })

            for director in movie_data["directors"]:
                self._create_person_relationship(
                    movie_data["tmdb_id"],
                    director["id"],
                    director["name"],
                    director.get("gender", 0),
                    director.get("birthday", ""),
                    "DIRECTED",
                    None
                )

            for actor in movie_data["cast"]:
                self._create_person_relationship(
                    movie_data["tmdb_id"],
                    actor["id"],
                    actor["name"],
                    actor.get("gender", 0),
                    actor.get("birthday", ""),
                    "ACTED_IN",
                    [actor.get("character", "")]
                )

            logger.info(f"Created movie: {movie_data['title']}")
            return True

        except Exception as e:
            logger.error(f"Failed to create movie {movie_id}: {e}")
            return False

    def _create_person_relationship(self, movie_id: int, person_id: int, name: str, 
                                  gender: int, birthday: str, 
                                  rel_type: str, roles: Optional[List[str]]) -> None:
        self.kg.query("""
        MERGE (p:Person {tmdb_id: $person_id})
        SET p.name = $name,
            p.gender = $gender,
            p.birthday = date($birthday)
        """, params={
            "person_id": person_id,
            "name": name,
            "gender": gender,
            "birthday": birthday if birthday else None
        })

        if rel_type == "ACTED_IN" and roles:
            self.kg.query("""
            MATCH (m:Movie {tmdb_id: $movie_id})
            MATCH (p:Person {tmdb_id: $person_id})
            MERGE (p)-[r:ACTED_IN]->(m)
            SET r.roles = $roles
            """, params={
                "movie_id": movie_id,
                "person_id": person_id,
                "roles": roles
            })
        else:
            self.kg.query(f"""
            MATCH (m:Movie {{tmdb_id: $movie_id}})
            MATCH (p:Person {{tmdb_id: $person_id}})
            MERGE (p)-[:{rel_type}]->(m)
            """, params={
                "movie_id": movie_id,
                "person_id": person_id
            })

    def import_popular_movies(self, count: int = 5) -> None:
        try:
            url = f"{self.base_url}/movie/popular"
            params = {
                "api_key": self.tmdb_api_key,
                "language": "en-US",
                "page": 1
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            movies = response.json()["results"][:count]
            
            for movie in movies:
                success = self.create_movie_from_tmdb(movie["id"])
                logger.info(f"Movie {movie['title']} imported: {'Success' if success else 'Failed'}")
                
        except Exception as e:
            logger.error(f"Failed to fetch popular movies: {e}")



    def close(self) -> None:
        self.kg.close()

def main():
    db = TMDBMovieDatabase()
    try:
        db.create_constraints()
        db.import_popular_movies(count=5)
        logger.info("Database population completed")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == '__main__':
    main()
