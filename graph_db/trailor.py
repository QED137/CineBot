import requests
from config import settings  # assumes TMDB_API_KEY is stored in .env or settings

TMDB_API_KEY = settings.TMDB_API_KEY
BASE_URL = "https://api.themoviedb.org/3"

def get_trailer_key(movie_id):
    """Fetch the YouTube trailer key for a movie by its TMDb ID."""
    url = f"{BASE_URL}/movie/{movie_id}/videos"
    params = {"api_key": TMDB_API_KEY}
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    videos = response.json().get("results", [])
    
    # Prioritize official YouTube trailers
    for video in videos:
        if video.get("site") == "YouTube" and video.get("type") == "Trailer" and video.get("official"):
            return video.get("key")

    # Fallback: any YouTube trailer
    for video in videos:
        if video.get("site") == "YouTube" and video.get("type") == "Trailer":
            return video.get("key")

    # Final fallback: any YouTube video
    for video in videos:
        if video.get("site") == "YouTube":
            return video.get("key")

    return None  # No suitable trailer found
movie_id = 27205  # Inception
trailer_key = get_trailer_key(movie_id)
if trailer_key:
    print("YouTube Trailer URL:", f"https://www.youtube.com/watch?v={trailer_key}")
else:
    print("No trailer found.")
