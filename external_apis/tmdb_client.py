import requests
from config import settings
import logging
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)

BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w185"

def get_tmdb_config():
    """Fetches TMDb configuration (like image base URLs) - cached ideally."""
    # TODO: Implement caching for this config
    if not settings.TMDB_API_KEY:
        log.warning("TMDB_API_KEY not set. Cannot fetch TMDb config.")
        return None
    try:
        url = f"{BASE_URL}/configuration"
        params = {"api_key": settings.TMDB_API_KEY}
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        log.error(f"Failed to fetch TMDb configuration: {e}")
        return None

def search_movie_tmdb(title: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Searches for a movie on TMDb by title and optional year."""
    if not settings.TMDB_API_KEY:
        log.debug("TMDB_API_KEY not set. Skipping TMDb search.")
        return None
    try:
        url = f"{BASE_URL}/search/movie"
        params = {
            "api_key": settings.TMDB_API_KEY,
            "query": title,
            "include_adult": False,
        }
        if year:
            params["primary_release_year"] = year # More specific search

        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            # Basic matching: return the first result
            # More sophisticated matching could compare year, director etc. if needed
            log.info(f"TMDb found match for '{title}' ({year if year else 'any year'}): {results[0].get('title')}")
            return results[0]
        else:
            log.info(f"TMDb search found no results for '{title}' ({year if year else 'any year'}).")
            return None

    except requests.exceptions.RequestException as e:
        log.error(f"TMDb API request failed for search '{title}': {e}")
        return None
    except Exception as e:
         log.error(f"An unexpected error occurred during TMDb search: {e}")
         return None

def get_poster_url(tmdb_movie_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extracts the full poster URL from TMDb movie data."""
    if tmdb_movie_data and tmdb_movie_data.get("poster_path"):
        poster_path = tmdb_movie_data["poster_path"]
        # You might want to fetch image base url from config dynamically
        # config = get_tmdb_config()
        # base_url = config['images']['secure_base_url']
        # size = config['images']['poster_sizes'][2] # Choose appropriate size
        # return f"{base_url}{size}{poster_path}"
        return f"{POSTER_BASE_URL}{poster_path}" # Using fixed base URL for simplicity
    return None

# Example usage
if __name__ == "__main__":
    movie_title = "Inception"
    movie_year = 2010

    if not settings.TMDB_API_KEY:
        print("Please set TMDB_API_KEY in your .env file to run this example.")
    else:
        tmdb_data = search_movie_tmdb(movie_title, movie_year)

        if tmdb_data:
            print(f"\nTMDb Data for {movie_title}:")
            print(f"  ID: {tmdb_data.get('id')}")
            print(f"  Title: {tmdb_data.get('title')}")
            print(f"  Overview: {tmdb_data.get('overview')[:100]}...") # Print start of overview
            poster = get_poster_url(tmdb_data)
            print(f"  Poster URL: {poster}")
        else:
            print(f"\nCould not find {movie_title} ({movie_year}) on TMDb.")