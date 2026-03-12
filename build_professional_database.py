#!/usr/bin/env python3
"""
Professional Movie Database Builder
Loads comprehensive movie data with:
- Full movie details (title, overview, release date, ratings, etc.)
- Genre relationships (HAS_GENRE)
- Person relationships (DIRECTED, ACTED_IN)
- Text embeddings for semantic search
- Poster embeddings for visual similarity
- Proper graph structure for complex queries
"""
import requests
import time
from config import settings
from graph_db.connection import get_driver, close_driver
import logging
from openai import OpenAI
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize clients
TMDB_API_KEY = settings.TMDB_API_KEY
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize CLIP for poster embeddings
logger.info("Loading CLIP model for poster embeddings...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def fetch_popular_movies(pages=10):
    """Fetch popular movies from TMDB (20 per page)"""
    all_movie_ids = set()
    
    logger.info(f"Fetching {pages} pages of popular movies...")
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/movie/popular"
        params = {"api_key": TMDB_API_KEY, "page": page}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for movie in data.get("results", []):
                movie_id = movie.get("id")
                if movie_id:
                    all_movie_ids.add(movie_id)
            
            logger.info(f"Page {page}/{pages}: {len(all_movie_ids)} unique movies so far")
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}")
    
    return list(all_movie_ids)

def fetch_movie_details(movie_id):
    """Fetch comprehensive movie details from TMDB"""
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,videos,keywords"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching details for movie {movie_id}: {e}")
        return None

def generate_text_embedding(text):
    """Generate text embedding using OpenAI"""
    if not text or len(text.strip()) == 0:
        return None
    
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text[:8000]  # Limit text length
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating text embedding: {e}")
        return None

def generate_poster_embedding(poster_path):
    """Generate poster embedding using CLIP"""
    if not poster_path:
        return None
    
    try:
        # Download poster image
        image_url = f"{IMAGE_BASE_URL}{poster_path}"
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Process image with CLIP
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            # Normalize embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features[0].cpu().numpy().tolist()
        
        return embedding
    except Exception as e:
        logger.error(f"Error generating poster embedding for {poster_path}: {e}")
        return None

def load_movie_to_neo4j(movie_data):
    """Load comprehensive movie data into Neo4j with all relationships and embeddings"""
    driver = get_driver()
    
    try:
        # Extract movie details
        tmdb_id = movie_data.get("id")
        title = movie_data.get("title", "Unknown")
        overview = movie_data.get("overview", "")
        release_date = movie_data.get("release_date", "")
        runtime = movie_data.get("runtime", 0)
        vote_average = movie_data.get("vote_average", 0.0)
        vote_count = movie_data.get("vote_count", 0)
        popularity = movie_data.get("popularity", 0.0)
        poster_path = movie_data.get("poster_path")
        backdrop_path = movie_data.get("backdrop_path")
        tagline = movie_data.get("tagline", "")
        budget = movie_data.get("budget", 0)
        revenue = movie_data.get("revenue", 0)
        
        # Generate embeddings
        logger.info(f"  Generating embeddings for '{title}'...")
        
        # Text embedding from overview
        text_for_embedding = f"{title}. {overview}"
        if tagline:
            text_for_embedding = f"{title}. {tagline}. {overview}"
        text_embedding = generate_text_embedding(text_for_embedding)
        
        # Poster embedding
        poster_embedding = generate_poster_embedding(poster_path)
        
        with driver.session(database="neo4j") as session:
            # Create/update Movie node
            movie_query = """
            MERGE (m:Movie {tmdb_id: $tmdb_id})
            SET m.title = $title,
                m.overview = $overview,
                m.release_date = $release_date,
                m.runtime = $runtime,
                m.vote_average = $vote_average,
                m.vote_count = $vote_count,
                m.popularity = $popularity,
                m.poster_path = $poster_path,
                m.backdrop_path = $backdrop_path,
                m.tagline = $tagline,
                m.budget = $budget,
                m.revenue = $revenue,
                m.taglineEmbedding = $text_embedding,
                m.posterEmbedding = $poster_embedding
            RETURN m.title AS title
            """
            
            session.run(movie_query, 
                tmdb_id=tmdb_id,
                title=title,
                overview=overview,
                release_date=release_date,
                runtime=runtime,
                vote_average=vote_average,
                vote_count=vote_count,
                popularity=popularity,
                poster_path=poster_path,
                backdrop_path=backdrop_path,
                tagline=tagline,
                budget=budget,
                revenue=revenue,
                text_embedding=text_embedding,
                poster_embedding=poster_embedding
            )
            
            # Add Genres
            genres = movie_data.get("genres", [])
            for genre in genres:
                genre_name = genre.get("name")
                if genre_name:
                    genre_query = """
                    MATCH (m:Movie {tmdb_id: $tmdb_id})
                    MERGE (g:Genre {name: $genre_name})
                    MERGE (m)-[:HAS_GENRE]->(g)
                    """
                    session.run(genre_query, tmdb_id=tmdb_id, genre_name=genre_name)
            
            # Add Director
            credits = movie_data.get("credits", {})
            crew = credits.get("crew", [])
            for person in crew:
                if person.get("job") == "Director":
                    director_name = person.get("name")
                    if director_name:
                        director_query = """
                        MATCH (m:Movie {tmdb_id: $tmdb_id})
                        MERGE (p:Person {name: $director_name})
                        SET p.profile_path = $profile_path
                        MERGE (p)-[:DIRECTED]->(m)
                        """
                        session.run(director_query, 
                            tmdb_id=tmdb_id,
                            director_name=director_name,
                            profile_path=person.get("profile_path")
                        )
            
            # Add Cast (top 10 actors)
            cast = credits.get("cast", [])[:10]
            for actor in cast:
                actor_name = actor.get("name")
                character = actor.get("character")
                if actor_name:
                    actor_query = """
                    MATCH (m:Movie {tmdb_id: $tmdb_id})
                    MERGE (p:Person {name: $actor_name})
                    SET p.profile_path = $profile_path
                    MERGE (p)-[r:ACTED_IN]->(m)
                    SET r.character = $character
                    """
                    session.run(actor_query,
                        tmdb_id=tmdb_id,
                        actor_name=actor_name,
                        character=character,
                        profile_path=actor.get("profile_path")
                    )
            
            logger.info(f"  [OK] Loaded '{title}' with {len(genres)} genres, director, and {len(cast)} actors")
            
    except Exception as e:
        logger.error(f"Error loading movie {movie_data.get('title', 'Unknown')}: {e}")
    
def create_vector_indexes():
    """Create vector indexes for semantic and visual search"""
    driver = get_driver()
    
    logger.info("Creating vector indexes...")
    
    with driver.session(database="neo4j") as session:
        # Text embedding index (1536 dimensions for OpenAI ada-002)
        try:
            session.run("""
                CREATE VECTOR INDEX movie_text_embeddings IF NOT EXISTS
                FOR (m:Movie)
                ON m.taglineEmbedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            logger.info("  [OK] Text embedding index created")
        except Exception as e:
            logger.warning(f"Text index already exists or error: {e}")
        
        # Poster embedding index (512 dimensions for CLIP)
        try:
            session.run("""
                CREATE VECTOR INDEX movie_poster_embeddings IF NOT EXISTS
                FOR (m:Movie)
                ON m.posterEmbedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 512,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            logger.info("  [OK] Poster embedding index created")
        except Exception as e:
            logger.warning(f"Poster index already exists or error: {e}")

def main():
    """Main function to build professional movie database"""
    
    logger.info("="*80)
    logger.info("BUILDING PROFESSIONAL MOVIE DATABASE")
    logger.info("="*80)
    
    # Step 1: Fetch movie IDs from TMDB
    logger.info("\n[1/4] Fetching popular movies from TMDB...")
    movie_ids = fetch_popular_movies(pages=10)  # 200 movies
    logger.info(f"Found {len(movie_ids)} unique movies to load\n")
    
    # Step 2: Load each movie with full details
    logger.info("[2/4] Loading movies with details, relationships, and embeddings...")
    total = len(movie_ids)
    
    for idx, movie_id in enumerate(movie_ids, 1):
        logger.info(f"[{idx}/{total}] Processing movie ID {movie_id}...")
        
        # Fetch full movie details
        movie_data = fetch_movie_details(movie_id)
        if movie_data:
            load_movie_to_neo4j(movie_data)
        
        # Rate limiting
        time.sleep(0.5)
        
        # Progress update every 20 movies
        if idx % 20 == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Progress: {idx}/{total} movies loaded ({idx/total*100:.1f}%)")
            logger.info(f"{'='*60}\n")
    
    # Step 3: Create vector indexes
    logger.info("\n[3/4] Creating vector indexes for semantic and visual search...")
    create_vector_indexes()
    
    # Step 4: Summary
    logger.info("\n[4/4] Database build complete!")
    logger.info("="*80)
    logger.info("DATABASE SUMMARY:")
    logger.info("  - Movies loaded with full details (title, overview, ratings, etc.)")
    logger.info("  - Genre relationships (HAS_GENRE) for genre-based queries")
    logger.info("  - Person relationships (DIRECTED, ACTED_IN) for factual queries")
    logger.info("  - Text embeddings for semantic search on descriptions")
    logger.info("  - Poster embeddings for visual similarity search")
    logger.info("="*80)
    logger.info("\nYour CineBot can now handle:")
    logger.info("  1. Poster-based search: Upload a poster to find similar movies")
    logger.info("  2. Factual queries: 'Who directed Titanic?', 'Who acted in...'")
    logger.info("  3. Genre queries: 'Tell me about romantic movies'")
    logger.info("  4. Semantic search: 'Movies about love and sacrifice'")
    logger.info("="*80)
    
    close_driver()

if __name__ == "__main__":
    main()
