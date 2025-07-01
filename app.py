# # app.py

# import os
# import io
# import re
# import random
# import logging
# from functools import lru_cache
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename
# from dotenv import load_dotenv
# from typing import List, Dict
# import html # FIXED: Removed duplicate import

# from core.core_rag import recommend_by_text, recommend_by_poster_image, logger
# from utils.poster_filter import is_valid_movie_poster

# load_dotenv()

# app = Flask(__name__, template_folder='templates', static_folder='static')
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# SUGGESTION_PROMPTS = [
#     "A gritty detective story set in a neon-drenched futuristic city.", "An uplifting animated film about an unlikely friendship.", "A mind-bending psychological thriller where reality is not what it seems.", "A hilarious comedy about a group of friends on a chaotic road trip.", "An epic historical drama about a forgotten leader who changed the world.", "A tense survival movie about being stranded in a remote wilderness.", "A charming romantic comedy with witty dialogue and a twist.", "A visually stunning fantasy adventure with dragons and ancient magic.",
# ]

# # --- Backend Helpers ---
# def parse_llm_recommendations(llm_text_response: str) -> List[Dict]:
#     recommendations = []
#     # This pattern is robust enough to handle the conversational output
#     pattern = re.compile(r"MOVIE:\s*(.*?)\s*\n\s*EXPLANATION:\s*(.*?)(?=\n\nMOVIE:|\Z)", re.DOTALL | re.IGNORECASE)
#     matches = pattern.findall(llm_text_response)

#     for match in matches:
#         title = match[0].strip()
#         explanation = match[1].strip()
#         recommendations.append({"title": title, "explanation": explanation})

#     # This handles cases where the LLM gives a conversational answer without recommendations
#     if not recommendations and llm_text_response and "I'm sorry" not in llm_text_response and "I couldn't find" not in llm_text_response:
#         logger.warning("LLM response was not in the expected structured format. Displaying as raw text.")
#         return [{"title": "CineBot's Thoughts", "explanation": llm_text_response}]
#     return recommendations

# def map_llm_recs_to_retrieved_details(
#     llm_parsed_recommendations: List[Dict],
#     initially_retrieved_movies: List[Dict]
# ) -> List[Dict]:
#     # If there are no retrieved movies (e.g., a follow-up question was answered), just return the parsed text.
#     if not initially_retrieved_movies:
#         logger.info("No new movies retrieved. Returning LLM text-only response.")
#         return llm_parsed_recommendations

#     detailed_recommendations = []
#     retrieved_lookup = {movie.get('title','').lower().strip(): movie for movie in initially_retrieved_movies}

#     for llm_rec in llm_parsed_recommendations:
#         llm_title_lower = llm_rec.get('title','').lower().strip()
#         matched_movie_data = retrieved_lookup.get(llm_title_lower)

#         if matched_movie_data:
#             detailed_rec = {
#                 "title": matched_movie_data.get('title'),
#                 "explanation": llm_rec.get('explanation'),
#                 "poster_url": matched_movie_data.get('poster_url'),
#                 "trailer_url": matched_movie_data.get('trailer_url'),
#                 "tagline": matched_movie_data.get('tagline'),
#                 "overview": matched_movie_data.get('overview'),
#                 "tmdb_id": matched_movie_data.get('tmdb_id')
#             }
#             detailed_recommendations.append(detailed_rec)
#         else:
#             logger.warning(f"Could not map LLM recommended title '{llm_rec.get('title')}' back to full details.")
#             detailed_recommendations.append({
#                 "title": llm_rec.get("title"),
#                 "explanation": llm_rec.get("explanation"),
#                 "poster_url": None,
#                 "trailer_url": None
#             })
#     return detailed_recommendations


# # --- HTML Rendering Helper ---
# def render_movie_card_html(rec: Dict, index: int) -> str:
#     poster_url = rec.get("poster_url") or 'https://via.placeholder.com/400x600.png?text=No+Poster'
#     title = html.escape(rec.get("title") or 'Recommendation')
#     explanation = html.escape(rec.get("explanation") or '...')
#     tmdb_id = rec.get("tmdb_id")
#     trailer_url = rec.get("trailer_url")

#     trailer_link = f'<a href="{trailer_url}" target="_blank" class="card-link">Trailer</a>' if trailer_url else ''
#     details_link = f'<a href="https://www.themoviedb.org/movie/{tmdb_id}" target="_blank" class="card-link">Details</a>' if tmdb_id else ''

#     return f"""
#     <div class="movie-card" style="animation-delay: {index * 100}ms;">
#         <div class="card-feedback">
#             <button class="feedback-btn" data-feedback="like" data-id="{tmdb_id}" title="Good rec!"><svg viewBox="0 0 24 24"><path fill="currentColor" d="M23,10C23,8.89 22.1,8 21,8H14.68L15.64,3.43C15.66,3.33 15.67,3.22 15.67,3.11C15.67,2.7 15.5,2.32 15.23,2.05L14.17,1L7.59,7.58C7.22,7.95 7,8.45 7,9V19A2,2 0 0,0 9,21H18C18.83,21 19.54,20.5 19.84,19.78L22.86,12.73C22.95,12.5 23,12.26 23,12V10Z"></path></svg></button>
#             <button class="feedback-btn" data-feedback="dislike" data-id="{tmdb_id}" title="Not what I wanted"><svg viewBox="0 0 24 24"><path fill="currentColor" d="M19,15H21A2,2 0 0,1 23,17V19A2,2 0 0,1 21,21H12.2C11.66,21 11.14,20.76 10.84,20.34L7.82,13.29C7.73,13.06 7.67,12.81 7.67,12.55V10.5A1.5,1.5 0 0,1 9.17,9L10.23,2.95C10.5,2.68 10.86,2.5 11.28,2.5C11.67,2.5 12.04,2.66 12.31,2.92L13.37,4L12.41,8.57C12.38,8.67 12.37,8.78 12.37,8.89V15H19M1,15H5V3H1V15Z"></path></svg></button>
#         </div>
#         <img src="{poster_url}" alt="Poster for {title}" class="poster-img">
#         <div class="card-content">
#             <h4>{title}</h4>
#             <div class="card-explanation"><p><i>CineBot says:</i> {explanation}</p></div>
#             <div class="card-actions">{trailer_link}{details_link}</div>
#         </div>
#     </div>
#     """

# # --- API Endpoints ---
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/suggestion', methods=['GET'])
# @lru_cache(maxsize=1)
# def get_suggestion():
#     return jsonify({"suggestion": random.choice(SUGGESTION_PROMPTS)})

# @app.route('/api/feedback', methods=['POST'])
# def handle_feedback():
#     data = request.get_json()
#     tmdb_id = data.get('tmdb_id')
#     feedback = data.get('feedback')
#     # Use a more detailed log message
#     logger.info(f"FEEDBACK RECEIVED: Movie ID='{tmdb_id}', Feedback='{feedback}'")
#     # Here you would typically store this feedback in a database
#     return jsonify({"status": "success", "message": f"Feedback received for {tmdb_id}"})


# # --- Recommendation Endpoints ---
# @app.route('/api/recommend/text', methods=['POST'])
# def handle_text_recommendation():
#     data = request.get_json()
#     if not data or 'query' not in data:
#         return jsonify({"error": "Missing 'query' in request body"}), 400
    
#     try:
#         query = data['query']
#         num_recs = data.get('num_recs', 3)
#         chat_history = data.get('history', [])
        
#         # This call correctly matches the conversational function in core_rag.py
#         llm_text, initial_movies = recommend_by_text(
#             user_query=query, 
#             chat_history=chat_history,
#             num_recommendations=num_recs
#         )
        
#         parsed_recs = parse_llm_recommendations(llm_text)
#         detailed_recs = map_llm_recs_to_retrieved_details(parsed_recs, initial_movies)
        
#         # If there are recommendations with details, render them as HTML cards
#         html_cards = ""
#         if any(rec.get("tmdb_id") for rec in detailed_recs):
#              html_cards = "".join([render_movie_card_html(rec, i) for i, rec in enumerate(detailed_recs)])
        
#         # The raw `llm_text` is also returned for the chat history and for pure text responses
#         return jsonify({"html": html_cards, "llm_response": llm_text})

#     except Exception as e:
#         logger.error(f"Error in text recommendation API: {e}", exc_info=True)
#         return jsonify({"error": "An internal error occurred. Please try again."}), 500

# @app.route('/api/recommend/image', methods=['POST'])
# def handle_image_recommendation():
#     if 'poster' not in request.files:
#         return jsonify({"error": "No 'poster' file part in the request"}), 400
#     file = request.files['poster']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400
#     try:
#         image_bytes = file.read()
#         # You might want to add more robust validation here
#         # if not is_valid_movie_poster(image_bytes):
#         #     return jsonify({"error": "Image does not appear to be a valid movie poster."}), 400
        
#         num_recs = int(request.form.get('num_recs', 3))
        
#         # This call now correctly matches the function signature in core_rag.py
#         llm_text, initial_movies = recommend_by_poster_image(
#             query_image_bytes=image_bytes, 
#             num_recommendations=num_recs
#         )
        
#         parsed_recs = parse_llm_recommendations(llm_text)
#         detailed_recs = map_llm_recs_to_retrieved_details(parsed_recs, initial_movies)

#         html_cards = "".join([render_movie_card_html(rec, i) for i, rec in enumerate(detailed_recs)])
        
#         return jsonify({"html": html_cards})

#     except Exception as e:
#         logger.error(f"Error in image recommendation API: {e}", exc_info=True)
#         return jsonify({"error": "CineBot encountered an internal error. Please try again."}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)

# app.py (Refactored for core_rag_refactored.py)

import os
import io
import re
import random
import logging
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import html

# --- IMPORTANT: Change this import ---
# from core.core_rag import recommend_by_text, recommend_by_poster_image, logger # OLD
from core.core_rag import process_query, logger # NEW

load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# --- NEW: Add a secret key for session management ---
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a-very-secret-key-for-dev")


SUGGESTION_PROMPTS = [
    # (Your list of prompts remains unchanged)
    "A gritty detective story set in a neon-drenched futuristic city.", "An uplifting animated film about an unlikely friendship.", "A mind-bending psychological thriller where reality is not what it seems.", "A hilarious comedy about a group of friends on a chaotic road trip.", "An epic historical drama about a forgotten leader who changed the world.", "A tense survival movie about being stranded in a remote wilderness.", "A charming romantic comedy with witty dialogue and a twist.", "A visually stunning fantasy adventure with dragons and ancient magic.",
]

# --- Backend Helpers (These are good, we'll keep them) ---
def parse_llm_recommendations(llm_text_response: str):
    # This function is fine as is.
    recommendations = []
    pattern = re.compile(r"MOVIE:\s*(.*?)\s*\n\s*EXPLANATION:\s*(.*?)(?=\n\nMOVIE:|\Z)", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(llm_text_response)

    # Handle direct answers from graph search or follow-ups that don't list movies
    if not matches and llm_text_response:
        # Check for common non-recommendation phrases
        non_rec_starters = ["who directed", "the director of", "was released", "stars", "is a movie"]
        if any(llm_text_response.lower().strip().startswith(s) for s in non_rec_starters):
             return [{"title": "CineBot's Answer", "explanation": llm_text_response}]
        # Fallback for other conversational text
        logger.warning("LLM response not in structured format. Displaying as raw text.")
        return [{"title": "CineBot's Thoughts", "explanation": llm_text_response}]

    for title, explanation in matches:
        recommendations.append({"title": title.strip(), "explanation": explanation.strip()})
    return recommendations

def map_llm_recs_to_retrieved_details(llm_parsed_recs, retrieved_movies):
    # This function is also fine as is.
    if not retrieved_movies:
        return llm_parsed_recs

    detailed_recs = []
    retrieved_lookup = {movie.get('title','').lower().strip(): movie for movie in retrieved_movies}

    for llm_rec in llm_parsed_recs:
        llm_title_lower = llm_rec.get('title','').lower().strip()
        matched_data = retrieved_lookup.get(llm_title_lower)
        if matched_data:
            detailed_recs.append({
                "title": matched_data.get('title'), "explanation": llm_rec.get('explanation'),
                "poster_url": matched_data.get('poster_url'), "trailer_url": matched_data.get('trailer_url'),
                "tagline": matched_data.get('tagline'), "overview": matched_data.get('overview'),
                "tmdb_id": matched_data.get('tmdb_id')
            })
        else:
            logger.warning(f"Could not map LLM title '{llm_rec.get('title')}' to details.")
            detailed_recs.append({ "title": llm_rec.get("title"), "explanation": llm_rec.get("explanation") })
    return detailed_recs

def render_movie_card_html(rec, index):
    # This function is also perfect.
    poster_url = rec.get("poster_url") or 'https://via.placeholder.com/400x600.png?text=No+Poster'
    title = html.escape(rec.get("title") or 'Recommendation')
    explanation = html.escape(rec.get("explanation") or '...')
    tmdb_id = rec.get("tmdb_id")
    trailer_url = rec.get("trailer_url")
    trailer_link = f'<a href="{trailer_url}" target="_blank" class="card-link">Trailer</a>' if trailer_url else ''
    details_link = f'<a href="https://www.themoviedb.org/movie/{tmdb_id}" target="_blank" class="card-link">Details</a>' if tmdb_id else ''

    # A check for non-movie cards (like a direct answer)
    if not tmdb_id and not trailer_url:
        return f"""
        <div class="movie-card text-only-card" style="animation-delay: {index * 100}ms;">
            <div class="card-content">
                <h4>{title}</h4>
                <div class="card-explanation"><p>{explanation}</p></div>
            </div>
        </div>
        """
    return f"""
    <div class="movie-card" style="animation-delay: {index * 100}ms;">
        <img src="{poster_url}" alt="Poster for {title}" class="poster-img">
        <div class="card-content">
            <h4>{title}</h4>
            <div class="card-explanation"><p><i>CineBot says:</i> {explanation}</p></div>
            <div class="card-actions">{trailer_link}{details_link}</div>
        </div>
    </div>
    """

# --- API Endpoints (These can stay as they are) ---
@app.route('/')
def index():
    session.clear() # Start a new session each time the page is visited
    return render_template('index.html')

@app.route('/api/suggestion', methods=['GET'])
def get_suggestion():
    return jsonify({"suggestion": random.choice(SUGGESTION_PROMPTS)})

@app.route('/api/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    logger.info(f"FEEDBACK RECEIVED: {data}")
    return jsonify({"status": "success"})


# --- NEW: UNIFIED CHAT ENDPOINT ---
# @app.route('/api/chat', methods=['POST'])
# def handle_chat():
#     try:
#         # Get chat history from server-side session
#         chat_history = session.get('chat_history', [])

#         # Get form data
#         user_query = request.form.get('query')
#         image_file = request.files.get('poster')
#         image_bytes = image_file.read() if image_file else None

#         # Add user's message to history
#         # We handle the display on the frontend, but log the user message here
#         if user_query:
#             chat_history.append({"role": "user", "content": user_query})
#         elif image_bytes:
#              chat_history.append({"role": "user", "content": "(Uploaded a poster)"})
#         else:
#             return jsonify({"error": "No query or image provided."}), 400

#         # --- THIS IS THE CORE CHANGE ---
#         # Call the single, unified process_query function
#         bot_response_text, context_movies = process_query(
#             user_query=user_query,
#             image_bytes=image_bytes,
#             chat_history=chat_history
#         )
#         # --------------------------------

#         # Add bot's response to history, including the crucial context for follow-ups
#         bot_message = {
#             "role": "assistant",
#             "content": bot_response_text,
#             "context": context_movies  # <-- This is the key for stateful conversation
#         }
#         chat_history.append(bot_message)

#         # Save the updated history back to the session
#         session['chat_history'] = chat_history

#         # Process the response to generate HTML cards for the frontend
#         parsed_recs = parse_llm_recommendations(bot_response_text)
#         detailed_recs = map_llm_recs_to_retrieved_details(parsed_recs, context_movies)
#         html_cards = "".join([render_movie_card_html(rec, i) for i, rec in enumerate(detailed_recs)])

#         # The raw text is sent for the chat bubble, the HTML is for the cards
#         return jsonify({
#             "llm_response_text": bot_response_text,
#             "html_cards": html_cards
#         })

#     except Exception as e:
#         logger.error(f"Error in chat API: {e}", exc_info=True)
#         return jsonify({"error": "An internal error occurred. Please try again."}), 500
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    import json
    try:
        # Try to get chat history from the request (preferred) or fallback to session
        history_json = request.form.get('chat_history')
        if history_json:
            chat_history = json.loads(history_json)
        else:
            chat_history = session.get('chat_history', [])

        # Extract user input
        user_query = request.form.get('query')
        image_file = request.files.get('poster')
        image_bytes = image_file.read() if image_file else None

        # Add user message to chat history
        if user_query:
            chat_history.append({"role": "user", "content": user_query})
        elif image_bytes:
            chat_history.append({"role": "user", "content": "(Uploaded a poster)"})
        else:
            return jsonify({"error": "No query or image provided."}), 400

        # 🔁 Core processing
        bot_response_text, context_movies = process_query(
            user_query=user_query,
            image_bytes=image_bytes,
            chat_history=chat_history
        )

        # Add assistant message with context
        bot_message = {
            "role": "assistant",
            "content": bot_response_text,
            "context": context_movies  # ✅ for follow-up queries
        }
        chat_history.append(bot_message)

        # Save updated history (optional if using client-side storage)
        session['chat_history'] = chat_history
        print("🧠 chat_history received:")
        for msg in chat_history:
            print(f"{msg['role']}: {msg['content'][:60]}")
            if msg["role"] == "assistant" and "context" in msg:
               print("  ↳ context: ✅ present")

        # Format response for frontend
        parsed_recs = parse_llm_recommendations(bot_response_text)
        detailed_recs = map_llm_recs_to_retrieved_details(parsed_recs, context_movies)
        html_cards = "".join([render_movie_card_html(rec, i) for i, rec in enumerate(detailed_recs)])
        
        return jsonify({
            "llm_response_text": bot_response_text,
            "html_cards": html_cards,
            "context_movies": context_movies  # 👈 return this so JS can store it
        })

    except Exception as e:
        logger.error(f"Error in chat API: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    