#!/usr/bin/env python3
"""Analyze the current database structure"""
from graph_db.connection import get_driver, close_driver
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

driver = get_driver()

with driver.session(database="neo4j") as session:
    print("\n" + "="*70)
    print("DATABASE STRUCTURE ANALYSIS")
    print("="*70)
    
    # Node counts by type
    print("\nNODE COUNTS:")
    result = session.run("""
        MATCH (n)
        RETURN labels(n)[0] AS label, count(*) AS count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"  {record['label']}: {record['count']:,}")
    
    # Total nodes
    result = session.run("MATCH (n) RETURN count(n) AS total")
    total_nodes = result.single()['total']
    print(f"\n  TOTAL NODES: {total_nodes:,}")
    
    # Relationship counts by type
    print("\nRELATIONSHIP COUNTS:")
    result = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"  {record['type']}: {record['count']:,}")
    
    # Total relationships
    result = session.run("MATCH ()-[r]->() RETURN count(r) AS total")
    total_rels = result.single()['total']
    print(f"\n  TOTAL RELATIONSHIPS: {total_rels:,}")
    
    # Movie statistics
    print("\n" + "="*70)
    print("MOVIE DATA COMPLETENESS:")
    print("="*70)
    
    # Movies with genres
    result = session.run("""
        MATCH (m:Movie)
        OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
        WITH m, count(g) AS genre_count
        RETURN 
            sum(CASE WHEN genre_count > 0 THEN 1 ELSE 0 END) AS with_genres,
            sum(CASE WHEN genre_count = 0 THEN 1 ELSE 0 END) AS without_genres
    """)
    rec = result.single()
    print(f"\nGenres:")
    print(f"  Movies WITH genres: {rec['with_genres']:,}")
    print(f"  Movies WITHOUT genres: {rec['without_genres']:,}")
    
    # Movies with directors
    result = session.run("""
        MATCH (m:Movie)
        RETURN 
            sum(CASE WHEN EXISTS((m)<-[:DIRECTED]-()) THEN 1 ELSE 0 END) AS with_director,
            sum(CASE WHEN NOT EXISTS((m)<-[:DIRECTED]-()) THEN 1 ELSE 0 END) AS without_director
    """)
    rec = result.single()
    print(f"\nDirectors:")
    print(f"  Movies WITH directors: {rec['with_director']:,}")
    print(f"  Movies WITHOUT directors: {rec['without_director']:,}")
    
    # Movies with actors
    result = session.run("""
        MATCH (m:Movie)
        RETURN 
            sum(CASE WHEN EXISTS((m)<-[:ACTED_IN]-()) THEN 1 ELSE 0 END) AS with_actors,
            sum(CASE WHEN NOT EXISTS((m)<-[:ACTED_IN]-()) THEN 1 ELSE 0 END) AS without_actors
    """)
    rec = result.single()
    print(f"\nActors:")
    print(f"  Movies WITH actors: {rec['with_actors']:,}")
    print(f"  Movies WITHOUT actors: {rec['without_actors']:,}")
    
    # Embeddings
    result = session.run("""
        MATCH (m:Movie)
        RETURN 
            sum(CASE WHEN m.taglineEmbedding IS NOT NULL AND m.taglineEmbedding <> [] THEN 1 ELSE 0 END) AS with_text_emb,
            sum(CASE WHEN m.posterEmbedding IS NOT NULL AND m.posterEmbedding <> [] THEN 1 ELSE 0 END) AS with_poster_emb,
            sum(CASE WHEN m.taglineEmbedding IS NULL OR m.taglineEmbedding = [] THEN 1 ELSE 0 END) AS without_text_emb,
            sum(CASE WHEN m.posterEmbedding IS NULL OR m.posterEmbedding = [] THEN 1 ELSE 0 END) AS without_poster_emb
    """)
    rec = result.single()
    print(f"\nText Embeddings:")
    print(f"  Movies WITH text embeddings: {rec['with_text_emb']:,}")
    print(f"  Movies WITHOUT text embeddings: {rec['without_text_emb']:,}")
    
    print(f"\nPoster Embeddings:")
    print(f"  Movies WITH poster embeddings: {rec['with_poster_emb']:,}")
    print(f"  Movies WITHOUT poster embeddings: {rec['without_poster_emb']:,}")
    
    # Check indexes
    print("\n" + "="*70)
    print("VECTOR INDEXES:")
    print("="*70)
    result = session.run("SHOW INDEXES")
    has_text_index = False
    has_poster_index = False
    for record in result:
        index_name = record.get('name', '')
        if 'text' in index_name.lower() or 'tagline' in index_name.lower():
            has_text_index = True
            print(f"  [OK] {index_name} (Text embeddings)")
        elif 'poster' in index_name.lower():
            has_poster_index = True
            print(f"  [OK] {index_name} (Poster embeddings)")
    
    if not has_text_index:
        print(f"  [FAIL] No text embedding index found")
    if not has_poster_index:
        print(f"  [FAIL] No poster embedding index found")
    
    # Sample movie
    print("\n" + "="*70)
    print("SAMPLE MOVIE DATA:")
    print("="*70)
    result = session.run("""
        MATCH (m:Movie)
        WHERE m.posterEmbedding IS NOT NULL
        RETURN m.title AS title, 
               m.overview AS overview,
               m.poster_path AS poster_path,
               EXISTS((m)-[:HAS_GENRE]->()) AS has_genre,
               EXISTS((m)<-[:DIRECTED]-()) AS has_director,
               EXISTS((m)<-[:ACTED_IN]-()) AS has_actors,
               m.taglineEmbedding IS NOT NULL AS has_text_emb,
               m.posterEmbedding IS NOT NULL AS has_poster_emb
        LIMIT 1
    """)
    rec = result.single()
    if rec:
        print(f"\nTitle: {rec['title']}")
        print(f"Has genres: {rec['has_genre']}")
        print(f"Has director: {rec['has_director']}")
        print(f"Has actors: {rec['has_actors']}")
        print(f"Has text embedding: {rec['has_text_emb']}")
        print(f"Has poster embedding: {rec['has_poster_emb']}")
    
    print("\n" + "="*70)

close_driver()
