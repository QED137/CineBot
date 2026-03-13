#!/usr/bin/env python3
"""
Simple script to load popular movies with genres from TMDB into Neo4j
"""
import requests
import time
from config import settings
from graph_db.connection import get_driver, close_driver
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TMDB_API_KEY = settings.TMDB_API_KEY
BASE_URL = "https://api.themoviedb.org/3"

def fetch_popular_movies(pages=5):
    """Fetch popular movies from TMDB"""
    all_movies = []
    
    for page in range(1, pages + 1):
        logger.info(f"Fetching page {page}/{pages}...")
        url = f"{BASE_URL}/movie/popular"
        params = {
            "api_key": TMDB_API_KEY,
            "page": page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for movie in data.get("results", []):
                movie_id = movie.get("id")
                if movie_id:
                    all_movies.append(movie_id)
            
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}")
    
    logger.info(f"Fetched {len(all_movies)} movie IDs")
    return all_movies

def fetch_movie_details(movie_id):
    """Fetch detailed movie info including genres"""
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,videos"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract trailer
        trailer_key = None
        videos = data.get("videos", {}).get("results", [])
        for video in videos:
            if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                trailer_key = video.get("key")
                break
        
        # Extract directors
        directors = []
        crew = data.get("credits", {}).get("crew", [])
        for person in crew:
            if person.get("job") == "Director":
                directors.append(person.get("name"))
        
        # Extract cast
        cast = []
        actors = data.get("credits", {}).get("cast", [])[:10]
        for actor in actors:
            cast.append(actor.get("name"))
        
        # Build movie info
        movie_info = {
            "tmdb_id": data.get("id"),
            "title": data.get("title"),
            "tagline": data.get("tagline", ""),
            "overview": data.get("overview", ""),
            "release_date": data.get("release_date"),
            "poster_path": data.get("poster_path", ""),
            "poster_url": f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get("poster_path") else "",
            "trailer_url": f"https://www.youtube.com/watch?v={trailer_key}" if trailer_key else "",
            "vote_average": data.get("vote_average", 0),
            "vote_count": data.get("vote_count", 0),
            "popularity": data.get("popularity", 0),
            "genres": [g.get("name") for g in data.get("genres", [])],
            "directors": directors,
            "cast": cast
        }
        
        time.sleep(0.3)  # Rate limiting
        return movie_info
        
    except Exception as e:
        logger.error(f"Error fetching movie {movie_id}: {e}")
        return None

def load_movie_to_neo4j(session, movie_data):
    """Load a single movie with genres, directors, and cast to Neo4j"""
    
    cypher = """
    MERGE (m:Movie {tmdb_id: $tmdb_id})
    ON CREATE SET
        m.title = $title,
        m.tagline = $tagline,
        m.overview = $overview,
        m.release_date = date($release_date),
        m.poster_path = $poster_path,
        m.poster_url = $poster_url,
        m.trailer_url = $trailer_url,
        m.vote_average = $vote_average,
        m.vote_count = $vote_count,
        m.popularity = $popularity,
        m.last_updated = datetime()
    ON MATCH SET
        m.title = $title,
        m.tagline = $tagline,
        m.overview = $overview,
        m.release_date = date($release_date),
        m.poster_path = $poster_path,
        m.poster_url = $poster_url,
        m.trailer_url = $trailer_url,
        m.vote_average = $vote_average,
        m.vote_count = $vote_count,
        m.popularity = $popularity,
        m.last_updated = datetime()
    
    // Create Genre nodes and relationships
    FOREACH (genre_name IN $genres |
        MERGE (g:Genre {name: genre_name})
        MERGE (m)-[:HAS_GENRE]->(g)
    )
    
    // Create Director nodes and relationships
    FOREACH (director_name IN $directors |
        MERGE (d:Person {name: director_name})
        MERGE (d)-[:DIRECTED]->(m)
    )
    
    // Create Actor nodes and relationships
    FOREACH (actor_name IN $cast |
        MERGE (a:Person {name: actor_name})
        MERGE (a)-[:ACTED_IN]->(m)
    )
    """
    
    try:
        # Handle null release_date
        release_date = movie_data.get("release_date")
        if not release_date or release_date == "":
            release_date = "2000-01-01"  # Default date for missing data
        
        params = {
            "tmdb_id": movie_data["tmdb_id"],
            "title": movie_data["title"],
            "tagline": movie_data["tagline"],
            "overview": movie_data["overview"],
            "release_date": release_date,
            "poster_path": movie_data["poster_path"],
            "poster_url": movie_data["poster_url"],
            "trailer_url": movie_data["trailer_url"],
            "vote_average": movie_data["vote_average"],
            "vote_count": movie_data["vote_count"],
            "popularity": movie_data["popularity"],
            "genres": movie_data["genres"],
            "directors": movie_data["directors"],
            "cast": movie_data["cast"]
        }
        
        session.run(cypher, params)
        return True
    except Exception as e:
        logger.error(f"Error loading movie {movie_data.get('title')}: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting movie data load...")
    
    # Create constraints first
    driver = get_driver()
    
    with driver.session(database="neo4j") as session:
        logger.info("Creating constraints...")
        constraints = [
            "CREATE CONSTRAINT movie_tmdb_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdb_id IS UNIQUE",
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.info(f"  Created: {constraint[:50]}...")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"  Already exists: {constraint[:50]}...")
                else:
                    logger.error(f"  Error: {e}")
    
    # Fetch movie IDs
    logger.info("\nFetching popular movies from TMDB...")
    movie_ids = fetch_popular_movies(pages=3)  # Start with 3 pages (60 movies)
    
    # Fetch and load each movie
    logger.info(f"\nLoading {len(movie_ids)} movies to Neo4j...")
    loaded_count = 0
    failed_count = 0
    
    with driver.session(database="neo4j") as session:
        for i, movie_id in enumerate(movie_ids, 1):
            logger.info(f"Processing {i}/{len(movie_ids)}: Movie ID {movie_id}")
            
            movie_data = fetch_movie_details(movie_id)
            if movie_data:
                if load_movie_to_neo4j(session, movie_data):
                    loaded_count += 1
                    logger.info(f"  [OK] Loaded: {movie_data['title']} (Genres: {', '.join(movie_data['genres'])})")
                else:
                    failed_count += 1
            else:
                failed_count += 1
            
            # Progress update every 10 movies
            if i % 10 == 0:
                logger.info(f"\n--- Progress: {loaded_count} loaded, {failed_count} failed ---\n")
    
    close_driver()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPLETE!")
    logger.info(f"Successfully loaded: {loaded_count} movies")
    logger.info(f"Failed: {failed_count} movies")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
