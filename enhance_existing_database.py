#!/usr/bin/env python3
"""
Enhance Existing Movie Database
This script fills in missing data for existing movies:
- Adds Genre relationships (HAS_GENRE)
- Adds Director relationships (DIRECTED)
- Adds Actor relationships (ACTED_IN)
- Adds missing text embeddings
- Adds poster embeddings for visual similarity
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

def fetch_movie_details_from_tmdb(tmdb_id):
    """Fetch comprehensive movie details from TMDB"""
    url = f"{BASE_URL}/movie/{tmdb_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching TMDB details for movie {tmdb_id}: {e}")
        return None

def generate_text_embedding(text):
    """Generate text embedding using OpenAI"""
    if not text or len(text.strip()) == 0:
        return None
    
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text[:8000]
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
        image_url = f"{IMAGE_BASE_URL}{poster_path}"
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features[0].cpu().numpy().tolist()
        
        return embedding
    except Exception as e:
        logger.error(f"Error generating poster embedding for {poster_path}: {e}")
        return None

def get_movies_needing_enhancement():
    """Get all movies and their current status"""
    driver = get_driver()
    
    with driver.session(database="neo4j") as session:
        query = """
        MATCH (m:Movie)
        OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
        WITH m, count(g) AS genre_count
        RETURN m.tmdb_id AS tmdb_id,
               m.title AS title,
               m.overview AS overview,
               m.tagline AS tagline,
               m.poster_path AS poster_path,
               m.taglineEmbedding AS text_embedding,
               m.posterEmbedding AS poster_embedding,
               genre_count,
               EXISTS((m)<-[:DIRECTED]-()) AS has_director,
               EXISTS((m)<-[:ACTED_IN]-()) AS has_actors
        ORDER BY genre_count ASC, m.title
        """
        result = session.run(query)
        return [dict(record) for record in result]

def enhance_movie(movie_info):
    """Enhance a single movie with missing data"""
    driver = get_driver()
    tmdb_id = movie_info['tmdb_id']
    title = movie_info['title']
    
    needs_update = False
    updates = []
    
    # Check what's missing
    if movie_info['genre_count'] == 0:
        needs_update = True
        updates.append("genres")
    
    if not movie_info['has_director']:
        needs_update = True
        updates.append("director")
    
    if not movie_info['has_actors']:
        needs_update = True
        updates.append("actors")
    
    if not movie_info['text_embedding']:
        needs_update = True
        updates.append("text embedding")
    
    if not movie_info['poster_embedding']:
        needs_update = True
        updates.append("poster embedding")
    
    if not needs_update:
        logger.info(f"  ✓ '{title}' - Already complete!")
        return False
    
    logger.info(f"  Enhancing '{title}' - Adding: {', '.join(updates)}")
    
    # Fetch TMDB data if we need genres/director/actors
    tmdb_data = None
    if 'genres' in updates or 'director' in updates or 'actors' in updates:
        tmdb_data = fetch_movie_details_from_tmdb(tmdb_id)
        if not tmdb_data:
            logger.warning(f"  ⚠ Could not fetch TMDB data for '{title}'")
            return False
        time.sleep(0.3)  # Rate limiting
    
    try:
        with driver.session(database="neo4j") as session:
            # Add Genres
            if 'genres' in updates and tmdb_data:
                genres = tmdb_data.get("genres", [])
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
            if 'director' in updates and tmdb_data:
                credits = tmdb_data.get("credits", {})
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
            
            # Add Actors
            if 'actors' in updates and tmdb_data:
                credits = tmdb_data.get("credits", {})
                cast = credits.get("cast", [])[:10]  # Top 10 actors
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
            
            # Add text embedding
            if 'text embedding' in updates:
                overview = movie_info.get('overview', '')
                tagline = movie_info.get('tagline', '')
                text = f"{title}. {tagline}. {overview}" if tagline else f"{title}. {overview}"
                
                text_embedding = generate_text_embedding(text)
                if text_embedding:
                    session.run("""
                        MATCH (m:Movie {tmdb_id: $tmdb_id})
                        SET m.taglineEmbedding = $embedding
                    """, tmdb_id=tmdb_id, embedding=text_embedding)
            
            # Add poster embedding
            if 'poster embedding' in updates:
                poster_path = movie_info.get('poster_path')
                if poster_path:
                    poster_embedding = generate_poster_embedding(poster_path)
                    if poster_embedding:
                        session.run("""
                            MATCH (m:Movie {tmdb_id: $tmdb_id})
                            SET m.posterEmbedding = $embedding
                        """, tmdb_id=tmdb_id, embedding=poster_embedding)
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Error enhancing '{title}': {e}")
        return False

def create_vector_indexes():
    """Create vector indexes for semantic and visual search"""
    driver = get_driver()
    
    logger.info("\nCreating/updating vector indexes...")
    
    with driver.session(database="neo4j") as session:
        # Text embedding index
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
            logger.info("  ✓ Text embedding index ready")
        except Exception as e:
            logger.info(f"  ✓ Text index exists")
        
        # Poster embedding index
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
            logger.info("  ✓ Poster embedding index ready")
        except Exception as e:
            logger.info(f"  ✓ Poster index exists")

def main():
    """Main function to enhance existing database"""
    
    logger.info("="*80)
    logger.info("ENHANCING EXISTING MOVIE DATABASE")
    logger.info("="*80)
    
    # Step 1: Get movies needing enhancement
    logger.info("\n[1/3] Analyzing database for missing data...")
    movies = get_movies_needing_enhancement()
    
    total_movies = len(movies)
    movies_needing_work = [m for m in movies if (
        m['genre_count'] == 0 or
        not m['has_director'] or
        not m['has_actors'] or
        not m['text_embedding'] or
        not m['poster_embedding']
    )]
    
    logger.info(f"Total movies: {total_movies}")
    logger.info(f"Movies needing enhancement: {len(movies_needing_work)}")
    logger.info(f"Movies already complete: {total_movies - len(movies_needing_work)}")
    
    if len(movies_needing_work) == 0:
        logger.info("\n✓ All movies are already complete!")
        close_driver()
        return
    
    # Step 2: Enhance movies
    logger.info(f"\n[2/3] Enhancing {len(movies_needing_work)} movies...")
    logger.info("This may take some time depending on the number of movies...\n")
    
    enhanced_count = 0
    for idx, movie in enumerate(movies_needing_work, 1):
        logger.info(f"[{idx}/{len(movies_needing_work)}] Processing...")
        if enhance_movie(movie):
            enhanced_count += 1
        
        # Progress update every 100 movies
        if idx % 100 == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Progress: {idx}/{len(movies_needing_work)} movies processed")
            logger.info(f"Successfully enhanced: {enhanced_count}")
            logger.info(f"{'='*60}\n")
    
    # Step 3: Create indexes
    create_vector_indexes()
    
    # Summary
    logger.info("\n[3/3] Enhancement complete!")
    logger.info("="*80)
    logger.info("SUMMARY:")
    logger.info(f"  Total movies in database: {total_movies}")
    logger.info(f"  Movies enhanced: {enhanced_count}")
    logger.info("="*80)
    logger.info("\nYour database now supports:")
    logger.info("  1. Poster-based search (visual similarity)")
    logger.info("  2. Factual queries (directors, actors)")
    logger.info("  3. Genre queries (HAS_GENRE relationships)")
    logger.info("  4. Semantic search (text embeddings)")
    logger.info("="*80)
    
    close_driver()

if __name__ == "__main__":
    main()
