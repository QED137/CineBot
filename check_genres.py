#!/usr/bin/env python3
"""Check what Genre nodes exist in the database"""
from graph_db.connection import get_driver, close_driver

driver = get_driver()

with driver.session(database="neo4j") as session:
    
    # Check if Genre nodes exist
    print("\n" + "="*80)
    print("Checking Genre nodes...")
    print("="*80)
    result = session.run("MATCH (g:Genre) RETURN g.name AS name LIMIT 50")
    records = list(result)
    
    if records:
        print(f"Found {len(records)} Genre nodes:")
        for record in records:
            print(f"  - {record['name']}")
    else:
        print("❌ No Genre nodes found!")
    
    # Check Movie-Genre relationships
    print("\n" + "="*80)
    print("Checking Movie-Genre relationships...")
    print("="*80)
    result = session.run("""
        MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre) 
        RETURN g.name AS genre, count(m) AS movie_count 
        ORDER BY movie_count DESC 
        LIMIT 20
    """)
    records = list(result)
    
    if records:
        print(f"Found {len(records)} genres with movies:")
        for record in records:
            print(f"  - {record['genre']}: {record['movie_count']} movies")
    else:
        print("❌ No Movie-Genre relationships found!")
    
    # Check a sample romantic movie
    print("\n" + "="*80)
    print("Looking for movies with 'Romance' genre...")
    print("="*80)
    result = session.run("""
        MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
        WHERE g.name CONTAINS 'Romance' OR g.name CONTAINS 'romance' OR g.name CONTAINS 'Romantic'
        RETURN m.title AS title, g.name AS genre
        LIMIT 5
    """)
    records = list(result)
    
    if records:
        print(f"Found {len(records)} movies:")
        for record in records:
            print(f"  - {record['title']} (Genre: {record['genre']})")
    else:
        print("❌ No movies found with Romance-related genre!")
        
        # Try case-insensitive search
        print("\nTrying case-insensitive search for any genre with 'rom' in it...")
        result = session.run("""
            MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
            WHERE toLower(g.name) CONTAINS 'rom'
            RETURN DISTINCT g.name AS genre
            LIMIT 10
        """)
        records = list(result)
        if records:
            print("Found genres:")
            for record in records:
                print(f"  - {record['genre']}")

close_driver()
