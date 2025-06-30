# app.py

import os
import io
import re
import random
import logging
from functools import lru_cache
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

# --- Your Core Logic Imports --- (with Mocks for robustness)
try:
    from core.core_rag import recommend_by_text, recommend_by_poster_image, logger
    from utils.poster_filter import is_valid_movie_poster
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    def mock_recommendation(query, top_k_retrieval=5, num_recommendations=3):
        mock_movies = [{"title": f"Mock Movie {i+1}", "explanation": "A great mock movie.", "poster_url": f"https://picsum.photos/seed/{random.randint(1,1000)}/400/600", "trailer_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "tagline": "An amazing tagline.", "overview": "A detailed overview.", "tmdb_id": f"mock_tmdb_{i+1}"} for i in range(num_recommendations)]
        llm_response = "\n\n".join([f"MOVIE: {m['title']}\nEXPLANATION: {m['explanation']}" for m in mock_movies])
        return llm_response, mock_movies
    def is_valid_movie_poster(image_bytes): return True
    recommend_by_text = recommend_by_poster_image = mock_recommendation
    logger.warning("Using MOCK RAG functions as core modules were not found.")

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

SUGGESTION_PROMPTS = [
    "A gritty detective story set in a neon-drenched futuristic city.", "An uplifting animated film about an unlikely friendship.", "A mind-bending psychological thriller where reality is not what it seems.", "A hilarious comedy about a group of friends on a chaotic road trip.", "An epic historical drama about a forgotten leader who changed the world.", "A tense survival movie about being stranded in a remote wilderness.", "A charming romantic comedy with witty dialogue and a twist.", "A visually stunning fantasy adventure with dragons and ancient magic.",
]

# --- Backend Helpers --- (unchanged from before)
def parse_llm_recommendations(llm_text_response: str) -> List[Dict]:
    recommendations = []
    pattern = re.compile(r"MOVIE:\s*(.*?)\s*\n\s*EXPLANATION:\s*(.*?)(?=\n\nMOVIE:|\Z)", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(llm_text_response)
    for match in matches:
        recommendations.append({"title": match[0].strip(), "explanation": match[1].strip()})
    if not recommendations and llm_text_response and "I'm sorry" not in llm_text_response:
        return [{"title": "CineBot's Thoughts", "explanation": llm_text_response}]
    return recommendations

def map_llm_recs_to_retrieved_details(llm_parsed: List[Dict], retrieved_movies: List[Dict]) -> List[Dict]:
    detailed_recommendations = []
    if not retrieved_movies: return [{"title": rec.get("title"), "explanation": rec.get("explanation")} for rec in llm_parsed]
    retrieved_lookup = {movie.get('title', '').lower().strip(): movie for movie in retrieved_movies}
    for llm_rec in llm_parsed:
        llm_title_lower = llm_rec.get('title', '').lower().strip()
        matched_movie = retrieved_lookup.get(llm_title_lower, {})
        detailed_rec = {**matched_movie, "explanation": llm_rec.get("explanation"), "title": matched_movie.get('title', llm_rec.get('title'))}
        detailed_recommendations.append(detailed_rec)
    return detailed_recommendations

# --- API Endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/suggestion', methods=['GET'])
@lru_cache(maxsize=1) # Cache the result to be light on the server
def get_suggestion():
    return jsonify({"suggestion": random.choice(SUGGESTION_PROMPTS)})

@app.route('/api/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    tmdb_id = data.get('tmdb_id')
    feedback = data.get('feedback')
    logger.info(f"FEEDBACK RECEIVED: Movie ID {tmdb_id} - Feedback: {feedback}")
    # In a real app, you would store this in your database.
    # For now, we just log it and confirm receipt.
    return jsonify({"status": "success", "message": f"Feedback received for {tmdb_id}"})

# ... (The rest of your recommendation endpoints are the same as before) ...
@app.route('/api/recommend/text', methods=['POST'])
def handle_text_recommendation():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    try:
        query = data['query']
        num_recs = data.get('num_recs', 3)
        llm_text, initial_movies = recommend_by_text(query, top_k_retrieval=5, num_recommendations=num_recs)
        parsed_recs = parse_llm_recommendations(llm_text)
        detailed_recs = map_llm_recs_to_retrieved_details(parsed_recs, initial_movies)
        return jsonify(detailed_recs)
    except Exception as e:
        logger.error(f"Error in text recommendation API: {e}", exc_info=True)
        return jsonify({"error": "CineBot encountered an internal error. Please try again."}), 500

@app.route('/api/recommend/image', methods=['POST'])
def handle_image_recommendation():
    if 'poster' not in request.files:
        return jsonify({"error": "No 'poster' file part in the request"}), 400
    file = request.files['poster']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        image_bytes = file.read()
        if not is_valid_movie_poster(image_bytes):
            return jsonify({"error": "Image does not appear to be a valid movie poster."}), 400
        
        num_recs = int(request.form.get('num_recs', 3))
        llm_text, initial_movies = recommend_by_poster_image(image_bytes, top_k_retrieval=5, num_recommendations=num_recs)
        parsed_recs = parse_llm_recommendations(llm_text)
        detailed_recs = map_llm_recs_to_retrieved_details(parsed_recs, initial_movies)
        return jsonify(detailed_recs)
    except Exception as e:
        logger.error(f"Error in image recommendation API: {e}", exc_info=True)
        return jsonify({"error": "CineBot encountered an internal error. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    