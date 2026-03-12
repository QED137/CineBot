#!/usr/bin/env python3
"""
Add Missing Data to Existing Database
Focuses on the two main gaps:
1. Add HAS_GENRE relationships (4,744 movies missing genres)
2. Add poster embeddings (4,799 movies missing poster embeddings)
3. Fill small gaps (16 text embeddings, 19 directors, 13 actors)
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
try:
    # Use HF token for authentication
    hf_token = getattr(settings, 'HF_TOKEN', None)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token=hf_token)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", token=hf_token)
    clip_model.eval()
    logger.info("[OK] CLIP model loaded\n")
except Exception as e:
    logger.error(f"Failed to load CLIP model: {e}")
    logger.info("Trying to import from core_rag instead...")
    # Try to import from core_rag if already loaded
    try:
        from core.core_rag import clip_model, clip_processor
        logger.info("[OK] Using CLIP model from core_rag\n")
    except Exception as e2:
        logger.error(f"Could not load CLIP model: {e2}")
        logger.error("Poster embeddings will be skipped!")
        clip_model = None
        clip_processor = None

def generate_poster_embedding(poster_path):
    """Generate poster embedding using CLIP"""
    if not poster_path or not clip_model or not clip_processor:
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
        logger.error(f"Error generating poster embedding: {e}")
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

def fetch_movie_genres_from_tmdb(tmdb_id):
    """Fetch only genre data from TMDB (lightweight)"""
    url = f"{BASE_URL}/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("genres", [])
    except Exception as e:
        logger.error(f"Error fetching genres for movie {tmdb_id}: {e}")
        return []

def add_poster_embeddings(batch_size=50):
    """Add poster embeddings to all movies"""
    logger.info("="*70)
    logger.info("STEP 1: Adding Poster Embeddings")
    logger.info("="*70)
    
    if not clip_model or not clip_processor:
        logger.warning("[WARNING] CLIP model not available. Skipping poster embeddings.")
        logger.info("You can add poster embeddings later when CLIP is available.\n")
        return
    
    driver = get_driver()
    
    with driver.session(database="neo4j") as session:
        # Get movies without poster embeddings
        result = session.run("""
            MATCH (m:Movie)
            WHERE (m.posterEmbedding IS NULL OR m.posterEmbedding = [])
            AND m.poster_path IS NOT NULL
            RETURN m.tmdb_id AS tmdb_id, 
                   m.title AS title,
                   m.poster_path AS poster_path
            ORDER BY m.popularity DESC
            LIMIT 5000
        """)
        movies = [dict(record) for record in result]
    
    total = len(movies)
    logger.info(f"Found {total:,} movies needing poster embeddings\n")
    
    if total == 0:
        logger.info("[OK] All movies already have poster embeddings!\n")
        return
    
    success_count = 0
    failed_count = 0
    
    for idx, movie in enumerate(movies, 1):
        title = movie['title']
        poster_path = movie['poster_path']
        
        # Generate poster embedding
        poster_embedding = generate_poster_embedding(poster_path)
        
        if poster_embedding:
            # Save to database
            with driver.session(database="neo4j") as session:
                session.run("""
                    MATCH (m:Movie {tmdb_id: $tmdb_id})
                    SET m.posterEmbedding = $embedding
                """, tmdb_id=movie['tmdb_id'], embedding=poster_embedding)
            
            success_count += 1
            if idx % 10 == 0:
                logger.info(f"  [{idx}/{total}] [OK] '{title}' - Success: {success_count}, Failed: {failed_count}")
        else:
            failed_count += 1
        
        # Progress update every 100 movies
        if idx % 100 == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            logger.info(f"Success: {success_count}, Failed: {failed_count}")
            logger.info(f"{'='*60}\n")
    
    logger.info(f"\n[OK] Poster embeddings complete: {success_count} added, {failed_count} failed\n")

def add_genre_relationships(batch_size=50):
    """Add HAS_GENRE relationships for movies without genres"""
    logger.info("="*70)
    logger.info("STEP 2: Adding Genre Relationships")
    logger.info("="*70)
    
    driver = get_driver()
    
    with driver.session(database="neo4j") as session:
        # Get movies without genres
        result = session.run("""
            MATCH (m:Movie)
            WHERE NOT EXISTS((m)-[:HAS_GENRE]->())
            RETURN m.tmdb_id AS tmdb_id, m.title AS title
            ORDER BY m.popularity DESC
            LIMIT 5000
        """)
        movies = [dict(record) for record in result]
    
    total = len(movies)
    logger.info(f"Found {total:,} movies needing genres\n")
    
    if total == 0:
        logger.info("[OK] All movies already have genres!\n")
        return
    
    success_count = 0
    failed_count = 0
    
    for idx, movie in enumerate(movies, 1):
        tmdb_id = movie['tmdb_id']
        title = movie['title']
        
        # Fetch genres from TMDB
        genres = fetch_movie_genres_from_tmdb(tmdb_id)
        
        if genres:
            # Add genre relationships
            with driver.session(database="neo4j") as session:
                for genre in genres:
                    genre_name = genre.get("name")
                    if genre_name:
                        session.run("""
                            MATCH (m:Movie {tmdb_id: $tmdb_id})
                            MERGE (g:Genre {name: $genre_name})
                            MERGE (m)-[:HAS_GENRE]->(g)
                        """, tmdb_id=tmdb_id, genre_name=genre_name)
            
            success_count += 1
            if idx % 10 == 0:
                genre_names = [g.get("name") for g in genres]
                logger.info(f"  [{idx}/{total}] [OK] '{title}' - {', '.join(genre_names)}")
        else:
            failed_count += 1
        
        # Rate limiting for TMDB API
        time.sleep(0.3)
        
        # Progress update every 100 movies
        if idx % 100 == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            logger.info(f"Success: {success_count}, Failed: {failed_count}")
            logger.info(f"{'='*60}\n")
    
    logger.info(f"\n[OK] Genre relationships complete: {success_count} movies updated, {failed_count} failed\n")

def fill_small_gaps():
    """Fill remaining small gaps (text embeddings, directors, actors)"""
    logger.info("="*70)
    logger.info("STEP 3: Filling Small Gaps")
    logger.info("="*70)
    
    driver = get_driver()
    
    # Add missing text embeddings
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (m:Movie)
            WHERE m.taglineEmbedding IS NULL OR m.taglineEmbedding = []
            RETURN m.tmdb_id AS tmdb_id, 
                   m.title AS title,
                   m.overview AS overview,
                   m.tagline AS tagline
            LIMIT 100
        """)
        movies = [dict(record) for record in result]
    
    if movies:
        logger.info(f"\nAdding text embeddings for {len(movies)} movies...")
        for movie in movies:
            title = movie['title']
            overview = movie.get('overview', '')
            tagline = movie.get('tagline', '')
            
            text = f"{title}. {tagline}. {overview}" if tagline else f"{title}. {overview}"
            embedding = generate_text_embedding(text)
            
            if embedding:
                with driver.session(database="neo4j") as session:
                    session.run("""
                        MATCH (m:Movie {tmdb_id: $tmdb_id})
                        SET m.taglineEmbedding = $embedding
                    """, tmdb_id=movie['tmdb_id'], embedding=embedding)
                logger.info(f"  [OK] '{title}'")
    else:
        logger.info("\n[OK] All movies already have text embeddings")
    
    logger.info("")

def main():
    """Main function"""
    
    logger.info("\n" + "="*70)
    logger.info("ENHANCING EXISTING DATABASE")
    logger.info("="*70 + "\n")
    
    # Step 1: Add poster embeddings (most important for visual search)
    add_poster_embeddings()
    
    # Step 2: Add genre relationships (needed for genre queries)
    add_genre_relationships()
    
    # Step 3: Fill small gaps
    fill_small_gaps()
    
    # Summary
    logger.info("="*70)
    logger.info("ENHANCEMENT COMPLETE!")
    logger.info("="*70)
    logger.info("\nYour database now supports:")
    logger.info("  1. [OK] Poster-based search (visual similarity)")
    logger.info("  2. [OK] Factual queries (directors, actors)")
    logger.info("  3. [OK] Genre queries (HAS_GENRE relationships)")
    logger.info("  4. [OK] Semantic search (text embeddings)")
    logger.info("="*70 + "\n")
    
    close_driver()

if __name__ == "__main__":
    main()
