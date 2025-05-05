import os
import requests
import logging
from typing import Optional, List, Dict
from urllib.parse import quote
from PIL import Image
import torch

from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from transformers import CLIPProcessor, CLIPModel

# --- Configuration ---
class Config:
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    OMDB_API = os.getenv("OMDB_API")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
    
    TMDB_BASE_URL = "https://api.themoviedb.org/3"
    OMDB_BASE_URL = f"http://www.omdbapi.com/?apikey={OMDB_API}&"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Global Models ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class MovieDatabase:
    def __init__(self):
        self.driver = self._connect_neo4j()
        
    def _connect_neo4j(self) -> Neo4jGraph:
        """Establish connection to Neo4j"""
        return Neo4jGraph(
            url=Config.NEO4J_URI,
            username=Config.NEO4J_USERNAME,
            password=Config.NEO4J_PASSWORD,
            database="neo4j"
        )
    
    def _execute_query(self, query: str, params: dict = None) -> List[Dict]:
        """Execute a Neo4j query with error handling"""
        try:
            return self.driver.query(query, params=params or {})
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Fetch complete movie details from TMDB API"""
        try:
            url = f"{Config.TMDB_BASE_URL}/movie/{movie_id}"
            params = {
                "api_key": Config.TMDB_API_KEY,
                "append_to_response": "credits"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "tmdb_id": data["id"],
                "title": data["title"],
                "tagline": data.get("tagline", ""),
                "overview": data.get("overview", ""),
                "release_date": data.get("release_date", ""),
                "poster_path": data.get("poster_path", ""),
                "directors": [
                    crew for crew in data["credits"]["crew"] 
                    if crew["job"] == "Director"
                ],
                "cast": data["credits"]["cast"][:10]  # Top 10 actors
            }
        except Exception as e:
            logger.error(f"Failed to fetch TMDB data: {e}")
            return None

    def create_movie_node(self, movie_data: Dict) -> bool:
        """Create a movie node with all relationships"""
        if not movie_data:
            return False

        # Create Movie node
        self._execute_query("""
        MERGE (m:Movie {tmdb_id: $tmdb_id})
        SET m.title = $title,
            m.tagline = $tagline,
            m.overview = $overview,
            m.release_date = date($release_date),
            m.poster_path = $poster_path
        """, params=movie_data)

        # Create Director relationships
        for director in movie_data["directors"]:
            self._execute_query("""
            MERGE (d:Person:Director {tmdb_id: $id})
            SET d.name = $name,
                d.gender = $gender,
                d.birthday = date($birthday)
            MERGE (d)-[:DIRECTED]->(m:Movie {tmdb_id: $movie_id})
            """, params={
                "id": director["id"],
                "name": director["name"],
                "gender": director.get("gender", 0),
                "birthday": director.get("birthday", ""),
                "movie_id": movie_data["tmdb_id"]
            })

        # Create Actor relationships
        for actor in movie_data["cast"]:
            self._execute_query("""
            MERGE (a:Person:Actor {tmdb_id: $id})
            SET a.name = $name,
                a.gender = $gender,
                a.birthday = date($birthday)
            MERGE (a)-[r:ACTED_IN]->(m:Movie {tmdb_id: $movie_id})
            SET r.character = $character,
                r.order = $order
            """, params={
                "id": actor["id"],
                "name": actor["name"],
                "gender": actor.get("gender", 0),
                "birthday": actor.get("birthday", ""),
                "movie_id": movie_data["tmdb_id"],
                "character": actor.get("character", ""),
                "order": actor.get("order", 99)
            })

        logger.info(f"Created nodes for: {movie_data['title']}")
        return True

    def import_popular_movies(self, limit: int = 20) -> int:
        """Import currently popular movies from TMDB"""
        try:
            url = f"{Config.TMDB_BASE_URL}/movie/popular"
            params = {
                "api_key": Config.TMDB_API_KEY,
                "language": "en-US",
                "page": 1
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            movies = response.json()["results"][:limit]
            
            success_count = 0
            for movie in movies:
                movie_data = self.get_movie_details(movie["id"])
                if movie_data and self.create_movie_node(movie_data):
                    success_count += 1
                    
            logger.info(f"Imported {success_count}/{len(movies)} movies")
            return success_count
        except Exception as e:
            logger.error(f"Failed to import movies: {e}")
            return 0

    def create_vector_indexes(self) -> None:
        """Create necessary vector indexes"""
        self._execute_query("""
        CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
        FOR (m:Movie) ON (m.taglineEmbedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
        """)
        
        self._execute_query("""
        CREATE VECTOR INDEX movie_poster_embeddings IF NOT EXISTS
        FOR (m:Movie) ON (m.posterEmbedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 512, `vector.similarity_function`: 'cosine'}}
        """)
        logger.info("Created vector indexes")

    def generate_tagline_embeddings(self) -> None:
        """Generate OpenAI embeddings for movie taglines"""
        self._execute_query("""
        MATCH (m:Movie) WHERE m.tagline IS NOT NULL AND m.taglineEmbedding IS NULL
        WITH m, genai.vector.encode(
            m.tagline, 
            "OpenAI", 
            {token: $apiKey, endpoint: $endpoint}
        ) AS embedding
        SET m.taglineEmbedding = embedding
        """, params={
            "apiKey": Config.OPENAI_API_KEY,
            "endpoint": Config.OPENAI_ENDPOINT
        })
        logger.info("Generated tagline embeddings")

    def generate_poster_embedding(self, image_url: str) -> Optional[List[float]]:
        """Generate CLIP embedding for a poster image"""
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")

            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embeddings = clip_model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            return None

    def update_movie_with_poster(self, title: str) -> bool:
        """Update movie with poster URL and embedding"""
        # Get poster URL from OMDB
        try:
            url = f"{Config.OMDB_BASE_URL}t={quote(title)}"
            response = requests.get(url, timeout=10)
            data = response.json()
            poster_url = data.get('Poster')
            if not poster_url:
                logger.warning(f"No poster found for {title}")
                return False
        except Exception as e:
            logger.error(f"OMDB API error: {e}")
            return False

        # Generate and store embedding
        embedding = self.generate_poster_embedding(poster_url)
        if not embedding:
            return False

        # Update Neo4j
        result = self._execute_query("""
        MATCH (m:Movie {title: $title})
        SET m.posterUrl = $url,
            m.posterEmbedding = $embedding
        RETURN m.title
        """, params={"title": title, "url": poster_url, "embedding": embedding})

        if result:
            logger.info(f"Updated poster for {title}")
            return True
        return False

    def get_trailer_key(self, movie_id: int) -> Optional[str]:
        """Fetch YouTube trailer key from TMDb"""
        try:
            url = f"{Config.TMDB_BASE_URL}/movie/{movie_id}/videos"
            params = {"api_key": Config.TMDB_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            videos = response.json().get("results", [])
            for video in videos:
                if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                    return video.get("key")
            return None
        except Exception as e:
            logger.error(f"TMDb API error: {e}")
            return None

    def print_movie_embeddings(self, title: str) -> None:
        """Print embeddings for a specific movie"""
        result = self._execute_query("""
        MATCH (m:Movie {title: $title})
        RETURN m.title AS title, 
               m.posterEmbedding AS poster_embedding,
               m.taglineEmbedding AS tagline_embedding
        LIMIT 1
        """, params={"title": title})

        if not result:
            logger.warning(f"Movie '{title}' not found")
            return

        movie = result[0]
        print(f"\n🎬 Movie: {movie['title']}")
        print("🖼️ Poster Embedding (first 5 dims):")
        print(movie['poster_embedding'][:5] if movie['poster_embedding'] else "Not available")
        print("💬 Tagline Embedding (first 5 dims):")
        print(movie['tagline_embedding'][:5] if movie['tagline_embedding'] else "Not available")

def main():
    db = MovieDatabase()
    
    # Initialize database
    db.create_vector_indexes()
    
    # Import popular movies
    db.import_popular_movies(limit=5)
    
    # Generate embeddings
    db.generate_tagline_embeddings()
    
    # Update posters for sample movies
    sample_movies = ["Inception", "The Dark Knight", "Interstellar"]
    for title in sample_movies:
        db.update_movie_with_poster(title)
    
    # Print sample embeddings
    db.print_movie_embeddings("Inception")
    
    # Get trailer for a movie
    trailer_key = db.get_trailer_key(27205)  # Inception
    if trailer_key:
        print(f"\n🎥 Trailer: https://www.youtube.com/watch?v={trailer_key}")

if __name__ == '__main__':
    main()