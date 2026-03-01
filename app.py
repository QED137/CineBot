import os
import io
import re
import random
import logging
import html
import json

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_session import Session

from core.core_rag import process_query, logger


# --- Load environment variables ---
load_dotenv()


# --- Flask app setup ---
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# Secret key for signing sessions (comes from /opt/cinebot/.env)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "INSECURE-DEV-SECRET")

# --- CORS Configuration for React Frontend ---
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# --- Server-side session config (Flask-Session) ---
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "/opt/cinebot/flask_session"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = False  # Set to True only when using HTTPS in production
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

Session(app)
# --------------------------------------------------


# --- Suggestion prompts for the UI ---
SUGGESTION_PROMPTS = [
    "A gritty detective story set in a neon-drenched futuristic city.",
    "An uplifting animated film about an unlikely friendship.",
    "A mind-bending psychological thriller where reality is not what it seems.",
    "A hilarious comedy about a group of friends on a chaotic road trip.",
    "An epic historical drama about a forgotten leader who changed the world.",
    "A tense survival movie about being stranded in a remote wilderness.",
    "A charming romantic comedy with witty dialogue and a twist.",
    "A visually stunning fantasy adventure with dragons and ancient magic.",
]


# --- Helpers to parse and render recommendations ---

def parse_llm_recommendations(llm_text_response: str):
    """
    Parse structured LLM output of the form:

        MOVIE: Title
        EXPLANATION: Some text

    Falls back to a single "CineBot's Thoughts" card if not structured.
    """
    recommendations = []
    pattern = re.compile(
        r"MOVIE:\s*(.*?)\s*\n\s*EXPLANATION:\s*(.*?)(?=\n\nMOVIE:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(llm_text_response)

    # Handle direct answers from graph search or plain Q&A
    if not matches and llm_text_response:
        non_rec_starters = [
            "who directed",
            "the director of",
            "was released",
            "stars",
            "is a movie",
        ]
        if any(llm_text_response.lower().strip().startswith(s) for s in non_rec_starters):
            return [{"title": "CineBot's Answer", "explanation": llm_text_response}]

        # Fallback: just show raw text in a card
        logger.warning("LLM response not in structured format. Displaying as raw text.")
        return [{"title": "CineBot's Thoughts", "explanation": llm_text_response}]

    for title, explanation in matches:
        recommendations.append(
            {
                "title": title.strip(),
                "explanation": explanation.strip(),
            }
        )
    return recommendations


def map_llm_recs_to_retrieved_details(llm_parsed_recs, retrieved_movies):
    """
    Map LLM-chosen titles back to the retrieved movie metadata (poster, trailer, etc.).
    """
    if not retrieved_movies:
        return llm_parsed_recs

    detailed_recs = []
    retrieved_lookup = {
        (movie.get("title") or "").lower().strip(): movie
        for movie in retrieved_movies
    }

    for llm_rec in llm_parsed_recs:
        llm_title_lower = (llm_rec.get("title") or "").lower().strip()
        matched_data = retrieved_lookup.get(llm_title_lower)

        if matched_data:
            detailed_recs.append(
                {
                    "title": matched_data.get("title"),
                    "explanation": llm_rec.get("explanation"),
                    "poster_url": matched_data.get("poster_url"),
                    "trailer_url": matched_data.get("trailer_url"),
                    "tagline": matched_data.get("tagline"),
                    "overview": matched_data.get("overview"),
                    "tmdb_id": matched_data.get("tmdb_id"),
                }
            )
        else:
            logger.warning(
                f"Could not map LLM title '{llm_rec.get('title')}' to details."
            )
            detailed_recs.append(
                {
                    "title": llm_rec.get("title"),
                    "explanation": llm_rec.get("explanation"),
                }
            )

    return detailed_recs


def render_movie_card_html(rec, index):
    """
    Turn a recommendation dict into a movie card HTML snippet.
    """
    poster_url = rec.get("poster_url") or "https://via.placeholder.com/400x600.png?text=No+Poster"
    title = html.escape(rec.get("title") or "Recommendation")
    explanation = html.escape(rec.get("explanation") or "...")
    tmdb_id = rec.get("tmdb_id")
    trailer_url = rec.get("trailer_url")

    trailer_link = (
        f'<a href="{trailer_url}" target="_blank" class="card-link">Trailer</a>'
        if trailer_url
        else ""
    )
    details_link = (
        f'<a href="https://www.themoviedb.org/movie/{tmdb_id}" target="_blank" class="card-link">Details</a>'
        if tmdb_id
        else ""
    )

    # Non-movie / pure text card (e.g., Q&A answer)
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


# --- Routes ---

@app.route("/")
def index():
    # Start with a fresh session each time the page is loaded
    session.clear()
    return render_template("index.html")


@app.route("/api/suggestion", methods=["GET"])
def get_suggestion():
    return jsonify({"suggestion": random.choice(SUGGESTION_PROMPTS)})


@app.route("/api/feedback", methods=["POST"])
def handle_feedback():
    data = request.get_json()
    logger.info(f"FEEDBACK RECEIVED: {data}")
    return jsonify({"status": "success"})


@app.route("/api/chat", methods=["POST"])
def handle_chat():
    """
    Unified chat endpoint:
    - Handles text queries (prompt box)
    - Handles poster uploads (image search)
    - Maintains a trimmed chat history in server-side session
    """
    try:
        MAX_TURNS = 6  # how many last messages to keep in server-side history

        # Prefer client-sent history (if you store it in JS), otherwise use session
        history_json = request.form.get("chat_history")
        if history_json:
            try:
                chat_history = json.loads(history_json)
            except json.JSONDecodeError:
                chat_history = []
        else:
            chat_history = session.get("chat_history", [])

        # Extract incoming data
        user_query = request.form.get("query")
        image_file = request.files.get("poster")
        image_bytes = image_file.read() if image_file else None

        # Basic validation: require either text or image
        if user_query:
            chat_history.append({"role": "user", "content": user_query})
        elif image_bytes:
            chat_history.append({"role": "user", "content": "(Uploaded a poster)"})
        else:
            return jsonify({"error": "No query or image provided."}), 400

        # Core RAG processing
        bot_response_text, context_movies = process_query(
            user_query=user_query,
            image_bytes=image_bytes,
            chat_history=chat_history,
        )

        # Add assistant message with full context (for the in-memory history / client)
        bot_message = {
            "role": "assistant",
            "content": bot_response_text,
            "context": context_movies,  # this can be heavy, so we won't store it in session
        }
        chat_history.append(bot_message)

        # --- Trim and store a LIGHT version of history in server-side session ---
        session_history = []
        for msg in chat_history[-MAX_TURNS:]:
            session_history.append(
                {
                    "role": msg.get("role"),
                    # store only text content in session, truncated
                    "content": (msg.get("content") or "")[:500],
                }
            )

        session["chat_history"] = session_history
        # -----------------------------------------------------------------------

        # Logging (optional, for debugging)
        print("🧠 chat_history (trimmed) stored in session:")
        for msg in session_history:
            print(f"{msg['role']}: {msg['content'][:60]}")

        # Build movie cards for UI
        parsed_recs = parse_llm_recommendations(bot_response_text)
        detailed_recs = map_llm_recs_to_retrieved_details(parsed_recs, context_movies)
        html_cards = "".join(
            [render_movie_card_html(rec, i) for i, rec in enumerate(detailed_recs)]
        )

        return jsonify(
            {
                "llm_response_text": bot_response_text,
                "html_cards": html_cards,
                "context_movies": context_movies,  # frontend can use this if needed
            }
        )

    except Exception as e:
        logger.error(f"Error in chat API: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred. Please try again."}), 500


if __name__ == "__main__":
    # For local debugging only; in production you use gunicorn
    app.run(debug=True, port=8000)

