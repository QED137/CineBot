#!/usr/bin/env python3
"""Check total movie count and genre status"""
from graph_db.connection import get_driver, close_driver
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

driver = get_driver()

with driver.session(database="neo4j") as session:
    # Total movies
    result = session.run("MATCH (m:Movie) RETURN count(m) AS total")
    total_movies = result.single()["total"]
    
    # Movies with genres
    result = session.run("""
        MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
        RETURN count(DISTINCT m) AS with_genres
    """)
    movies_with_genres = result.single()["with_genres"]
    
    # Movies with embeddings
    result = session.run("""
        MATCH (m:Movie)
        WHERE m.taglineEmbedding IS NOT NULL AND m.taglineEmbedding <> []
        RETURN count(m) AS with_embeddings
    """)
    movies_with_embeddings = result.single()["with_embeddings"]
    
    print(f"\n{'='*60}")
    print(f"Total movies in database: {total_movies}")
    print(f"Movies with genres: {movies_with_genres}")
    print(f"Movies without genres: {total_movies - movies_with_genres}")
    print(f"Movies with embeddings: {movies_with_embeddings}")
    print(f"Movies without embeddings: {total_movies - movies_with_embeddings}")
    print(f"{'='*60}\n")

close_driver()
