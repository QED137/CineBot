#!/usr/bin/env python3
"""
Script to create indexes on Neo4j database for better query performance.
Run this script to optimize your CineBot database.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from neo4j import GraphDatabase

def create_indexes():
    """Create indexes on frequently queried properties"""
    
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )
    
    indexes = [
        # Index on Genre name for fast genre lookups
        "CREATE INDEX genre_name_index IF NOT EXISTS FOR (g:Genre) ON (g.name)",
        
        # Index on Movie tmdb_id for fast movie lookups
        "CREATE INDEX movie_tmdb_id_index IF NOT EXISTS FOR (m:Movie) ON (m.tmdb_id)",
        
        # Index on Movie title for text searches
        "CREATE INDEX movie_title_index IF NOT EXISTS FOR (m:Movie) ON (m.title)",
        
        # Composite index on vote_average and popularity for sorting
        "CREATE INDEX movie_vote_popularity_index IF NOT EXISTS FOR (m:Movie) ON (m.vote_average, m.popularity)",
        
        # Index on Person name for director/actor searches
        "CREATE INDEX person_name_index IF NOT EXISTS FOR (p:Person) ON (p.name)",
    ]
    
    try:
        with driver.session() as session:
            print("[CONFIG] Creating indexes on Neo4j database...")
            print("=" * 60)
            
            for index_query in indexes:
                try:
                    session.run(index_query)
                    index_name = index_query.split("FOR")[0].split("IF NOT EXISTS")[0].strip()
                    print(f"[OK] {index_name}")
                except Exception as e:
                    print(f"[WARNING]  Failed to create index: {e}")
            
            print("=" * 60)
            print("\n[STATS] Checking existing indexes...")
            result = session.run("SHOW INDEXES")
            for record in result:
                print(f"  - {record.get('name', 'N/A')}: {record.get('labelsOrTypes', 'N/A')} ON {record.get('properties', 'N/A')}")
            
            print("\n[OK] Index creation complete!")
            print("\n[TIP] Tip: Indexes will improve query performance, especially for:")
            print("   • Genre-based searches")
            print("   • Movie lookups by title or ID")
            print("   • Sorting by ratings and popularity")
            print("   • Director and actor searches")
            
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()


if __name__ == "__main__":
    print("CineBot Database Index Creator")
    print("=" * 60)
    create_indexes()
