#!/usr/bin/env python3
"""
Quick script to inspect Neo4j database schema and data
"""
from graph_db.connection import get_driver, close_driver
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_query(session, query, description):
    """Helper to run query and print results"""
    print(f"\n{'='*80}")
    print(f"Query: {description}")
    print(f"{'='*80}")
    try:
        result = session.run(query)
        records = list(result)
        if records:
            for record in records:
                print(record.data())
        else:
            print("No results found.")
        return records
    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    driver = get_driver()
    
    with driver.session(database="neo4j") as session:
        
        # 1. Check what labels exist
        run_query(session, 
                 "CALL db.labels()",
                 "All node labels in database")
        
        # 2. Check Genre node properties
        run_query(session, 
                 "MATCH (g:Genre) RETURN g LIMIT 5",
                 "Sample Genre nodes")
        
        # 3. Check Genre node property keys
        run_query(session, 
                 "MATCH (g:Genre) RETURN DISTINCT keys(g) AS properties LIMIT 1",
                 "Genre node properties")
        
        # 4. Check Movie node property keys
        run_query(session, 
                 "MATCH (m:Movie) RETURN DISTINCT keys(m) AS properties LIMIT 1",
                 "Movie node properties")
        
        # 5. Check if any Genre nodes have embeddings
        run_query(session, 
                 """MATCH (g:Genre) 
                    WHERE g.embedding IS NOT NULL OR g.genreEmbedding IS NOT NULL
                    RETURN g.name, keys(g) AS properties LIMIT 5""",
                 "Genre nodes with embeddings")
        
        # 6. Check if any Movie nodes have genre-related embeddings
        run_query(session, 
                 """MATCH (m:Movie) 
                    WHERE m.genreEmbedding IS NOT NULL
                    RETURN m.title, keys(m) AS properties LIMIT 5""",
                 "Movie nodes with genreEmbedding")
        
        # 7. List all vector indexes
        run_query(session, 
                 "SHOW INDEXES YIELD name, type, labelsOrTypes, properties WHERE type = 'VECTOR' RETURN name, labelsOrTypes, properties",
                 "All vector indexes")
        
        # 8. Check Movie-Genre relationships
        run_query(session, 
                 """MATCH (m:Movie)-[r:HAS_GENRE]->(g:Genre) 
                    RETURN m.title, g.name LIMIT 10""",
                 "Movie-Genre relationships (sample)")
        
        # 9. Count nodes
        run_query(session, 
                 "MATCH (m:Movie) RETURN count(m) AS movie_count",
                 "Total Movie count")
        
        run_query(session, 
                 "MATCH (g:Genre) RETURN count(g) AS genre_count",
                 "Total Genre count")
        
        # 10. Check a specific movie with all its properties
        run_query(session,
                 """MATCH (m:Movie) 
                    WHERE m.title CONTAINS 'Titanic'
                    RETURN m""",
                 "Titanic movie with all properties")
        
        # 11. Check if Genre nodes have explanation property
        run_query(session,
                 """MATCH (g:Genre)
                    RETURN g.name AS name, 
                           CASE WHEN g.explanation IS NOT NULL THEN 'YES' ELSE 'NO' END AS has_explanation,
                           CASE WHEN g.embedding IS NOT NULL THEN 'YES' ELSE 'NO' END AS has_embedding
                    LIMIT 20""",
                 "Genre nodes - checking for explanation and embedding properties")

    close_driver()

if __name__ == "__main__":
    main()
