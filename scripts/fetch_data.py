import requests
#from config import settings
from graph_db.connection import get_driver, close_driver
from embeddings.generator import generate_embedding, get_embedding_model
from config import settings # Ensure settings loads TMDB_API_KEY
import requests

# Replace this with your TMDb API key
API_KEY = settings.TMDB_API_KEY
BASE_URL = "https://api.themoviedb.org/3"

def get_movie_by_title(title):
    search_url = f"{BASE_URL}/search/movie"
    params = {
        "api_key": API_KEY,
        "query": title
    }
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    results = response.json()["results"]

    if not results:
        print("No results found.")
        return None

    # Return the first matching movie ID
    return results[0]["id"]

def get_movie_details(movie_id):
    movie_url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": API_KEY
    }
    response = requests.get(movie_url, params=params)
    response.raise_for_status()
    return response.json()
def get_actors_detail(movie_json, top_n=5):
    cast = movie_json.get("credits", {}).get("cast", [])
    return [actor["name"] for actor in cast[:top_n]]

# Example usage
if __name__ == "__main__":
    movie_title = input("Enter movie title: ")
    movie_id = get_movie_by_title(movie_title)

    if movie_id:
        details = get_movie_details(movie_id)
        print(f"\n🎬 Title: {details.get('title')}")
        print(f"🗓️ Release Date: {details.get('release_date')}")
        print(f"🧾 Tagline: {details.get('tagline')}")
        print(f"📖 Overview: {details.get('overview')}")
        print(f"🎭 Genres: {[genre['name'] for genre in details.get('genres', [])]}")
        print(f"🖼️ Poster URL: https://image.tmdb.org/t/p/w500{details.get('poster_path')}")
