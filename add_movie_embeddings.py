#!/usr/bin/env python3
"""
Add text embeddings to movie overviews in Neo4j
"""
from graph_db.connection import get_driver, close_driver
from config import settings
import logging
import time
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_text_embedding(text: str):
    """Generate text embedding using OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def add_embeddings_to_movies():
    """Add text embeddings to all movies that don't have them"""
    
    driver = get_driver()
    
    with driver.session(database="neo4j") as session:
        # Get count of movies without embeddings
        count_query = """
        MATCH (m:Movie)
        WHERE m.taglineEmbedding IS NULL OR m.taglineEmbedding = []
        RETURN count(m) AS count
        """
        result = session.run(count_query)
        total = result.single()["count"]
        
        if total == 0:
            logger.info("All movies already have embeddings!")
            close_driver()
            return
        
        logger.info(f"Found {total} movies without embeddings")
        
        # Fetch movies needing embeddings in batches
        fetch_query = """
        MATCH (m:Movie)
        WHERE m.taglineEmbedding IS NULL OR m.taglineEmbedding = []
        RETURN m.tmdb_id AS tmdb_id, m.title AS title, 
               m.tagline AS tagline, m.overview AS overview
        LIMIT 50
        """
        
        processed = 0
        
        while True:
            result = session.run(fetch_query)
            movies = list(result)
            
            if not movies:
                break
            
            logger.info(f"\nProcessing batch of {len(movies)} movies...")
            
            for movie in movies:
                tmdb_id = movie["tmdb_id"]
                title = movie["title"]
                tagline = movie["tagline"] or ""
                overview = movie["overview"] or ""
                
                # Create text to embed (tagline + overview)
                text_to_embed = f"{tagline} {overview}".strip()
                
                if not text_to_embed:
                    logger.warning(f"  [WARNING]  {title}: No text to embed, skipping")
                    continue
                
                # Generate embedding
                try:
                    embedding = generate_text_embedding(text_to_embed)
                    
                    if embedding:
                        # Store embedding
                        update_query = """
                        MATCH (m:Movie {tmdb_id: $tmdb_id})
                        SET m.taglineEmbedding = $embedding
                        """
                        session.run(update_query, {"tmdb_id": tmdb_id, "embedding": embedding})
                        
                        processed += 1
                        logger.info(f"  [OK] {processed}/{total}: {title}")
                    else:
                        logger.error(f"  [ERROR] {title}: Failed to generate embedding")
                        
                except Exception as e:
                    logger.error(f"  [ERROR] {title}: Error - {e}")
                
                # Rate limiting for OpenAI API
                time.sleep(0.1)
            
            logger.info(f"\n--- Progress: {processed}/{total} movies processed ---\n")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETE! Processed {processed} movies")
        logger.info(f"{'='*80}")
    
    close_driver()

def create_vector_index():
    """Create vector index for movie embeddings"""
    driver = get_driver()
    
    with driver.session(database="neo4j") as session:
        logger.info("\nCreating vector index...")
        
        index_query = """
        CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
        FOR (m:Movie) ON (m.taglineEmbedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """
        
        try:
            session.run(index_query)
            logger.info("[OK] Vector index created successfully")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("[OK] Vector index already exists")
            else:
                logger.error(f"[ERROR] Error creating index: {e}")
    
    close_driver()

if __name__ == "__main__":
    logger.info("Starting embedding generation...\n")
    
    # Step 1: Create vector index
    create_vector_index()
    
    # Step 2: Add embeddings to movies
    add_embeddings_to_movies()
    
    logger.info("\nAll done! Your movies now have text embeddings for vector search.")
