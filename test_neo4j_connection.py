#!/usr/bin/env python3
"""
Quick diagnostic script to check Neo4j connection and query performance.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from neo4j import GraphDatabase

def test_connection():
    """Test Neo4j connection and query performance"""
    
    print("🔍 CineBot Neo4j Connection Diagnostic")
    print("=" * 60)
    
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        connection_timeout=10,
        max_connection_lifetime=300
    )
    
    try:
        with driver.session() as session:
            # Test 1: Basic connection
            print("\n1️⃣ Testing basic connection...")
            start = time.time()
            result = session.run("RETURN 1 as test")
            result.single()
            elapsed = time.time() - start
            print(f"   ✅ Connection OK ({elapsed:.2f}s)")
            
            # Test 2: Count nodes
            print("\n2️⃣ Counting database nodes...")
            start = time.time()
            result = session.run("""
                MATCH (n)
                RETURN 
                    count(CASE WHEN n:Movie THEN 1 END) as movies,
                    count(CASE WHEN n:Genre THEN 1 END) as genres,
                    count(CASE WHEN n:Person THEN 1 END) as people
            """)
            counts = result.single()
            elapsed = time.time() - start
            print(f"   Movies: {counts['movies']}")
            print(f"   Genres: {counts['genres']}")
            print(f"   People: {counts['people']}")
            print(f"   ⏱️  Query time: {elapsed:.2f}s")
            
            # Test 3: Genre query (the one that's failing)
            print("\n3️⃣ Testing genre-based search (Science Fiction)...")
            start = time.time()
            result = session.run("""
                MATCH (g:Genre {name: $genre_name})<-[:HAS_GENRE]-(m:Movie)
                WHERE m.vote_average IS NOT NULL AND m.popularity IS NOT NULL
                RETURN count(m) as movie_count
                LIMIT 1
            """, genre_name="Science Fiction")
            count = result.single()['movie_count']
            elapsed = time.time() - start
            print(f"   Found {count} Science Fiction movies")
            print(f"   ⏱️  Query time: {elapsed:.2f}s")
            
            if elapsed > 5:
                print(f"   ⚠️  SLOW! This query should be < 1s. Consider creating indexes:")
                print(f"      python create_indexes.py")
            
            # Test 4: Check indexes
            print("\n4️⃣ Checking database indexes...")
            result = session.run("SHOW INDEXES")
            indexes = list(result)
            if indexes:
                print(f"   Found {len(indexes)} indexes:")
                for idx in indexes:
                    print(f"   - {idx.get('name', 'N/A')}")
            else:
                print("   ⚠️  No indexes found! Run: python create_indexes.py")
            
            # Test 5: Sample movies
            print("\n5️⃣ Sample movies in database...")
            result = session.run("""
                MATCH (m:Movie)
                RETURN m.title as title, m.vote_average as rating
                ORDER BY m.popularity DESC
                LIMIT 5
            """)
            for i, movie in enumerate(result, 1):
                print(f"   {i}. {movie['title']} (Rating: {movie['rating']})")
            
            print("\n" + "=" * 60)
            print("✅ Diagnostic complete!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()


if __name__ == "__main__":
    test_connection()
