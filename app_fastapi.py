import os
import io
import re
import random
import logging
import html
import json
import time
import asyncio
from collections import defaultdict, deque
from threading import Lock
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

from core.core_rag import process_query, logger


# --- Load environment variables ---
load_dotenv()


# --- FastAPI app setup ---
app = FastAPI(title="CineBot API", version="2.0.0")

# Secret key for signing sessions
SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
if not SECRET_KEY:
    if os.getenv("FLASK_ENV", "").lower() == "production":
        raise RuntimeError("FLASK_SECRET_KEY must be set in production.")
    SECRET_KEY = os.urandom(32).hex()
    logger.warning("FLASK_SECRET_KEY is missing. Using an ephemeral dev secret key.")

# Session middleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# --- CORS Configuration for React Frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Templates directory
templates = Jinja2Templates(directory="templates")


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%s, using default=%s", name, raw, default)
        return default


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%s, using default=%s", name, raw, default)
        return default


MAX_QUERY_CHARS = _get_int_env("MAX_QUERY_CHARS", 500)
SESSION_REQUEST_LIMIT = _get_int_env("SESSION_REQUEST_LIMIT", 10)
SESSION_WINDOW_SECONDS = _get_int_env("SESSION_WINDOW_SECONDS", 300)
SESSION_COST_BUDGET = _get_int_env("SESSION_COST_BUDGET", 30)
SESSION_COST_WINDOW_SECONDS = _get_int_env("SESSION_COST_WINDOW_SECONDS", 3600)
SESSION_MIN_INTERVAL_SECONDS = _get_float_env("SESSION_MIN_INTERVAL_SECONDS", 0.75)
TEXT_REQUEST_COST = _get_int_env("TEXT_REQUEST_COST", 3)
IMAGE_REQUEST_COST = _get_int_env("IMAGE_REQUEST_COST", 8)
GLOBAL_CHAT_CONCURRENCY = _get_int_env("GLOBAL_CHAT_CONCURRENCY", 30)

CHAT_RATE_LIMIT = _get_int_env("CHAT_RATE_LIMIT", 20)
CHAT_RATE_WINDOW_SECONDS = _get_int_env("CHAT_RATE_WINDOW_SECONDS", 60)
CHAT_BURST_LIMIT = _get_int_env("CHAT_BURST_LIMIT", 4)
CHAT_BURST_WINDOW_SECONDS = _get_int_env("CHAT_BURST_WINDOW_SECONDS", 10)
SUGGESTION_RATE_LIMIT = _get_int_env("SUGGESTION_RATE_LIMIT", 60)
SUGGESTION_RATE_WINDOW_SECONDS = _get_int_env("SUGGESTION_RATE_WINDOW_SECONDS", 60)
FEEDBACK_RATE_LIMIT = _get_int_env("FEEDBACK_RATE_LIMIT", 30)
FEEDBACK_RATE_WINDOW_SECONDS = _get_int_env("FEEDBACK_RATE_WINDOW_SECONDS", 60)

chat_concurrency_guard = asyncio.Semaphore(max(1, GLOBAL_CHAT_CONCURRENCY))
_ip_window_events = defaultdict(deque)
_ip_window_lock = Lock()


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_ip_limit(request: Request, scope: str, limit: int, window_seconds: int):
    now = time.time()
    ip = _client_ip(request)
    key = f"{scope}:{ip}:{window_seconds}"
    with _ip_window_lock:
        events = _ip_window_events[key]
        while events and now - events[0] > window_seconds:
            events.popleft()
        if len(events) >= limit:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded for {scope}.")
        events.append(now)


def _enforce_session_limits(request: Request, is_image: bool):
    now = time.time()
    guard = request.session.get("abuse_guard", {})

    last_ts = float(guard.get("last_ts", 0))
    if now - last_ts < SESSION_MIN_INTERVAL_SECONDS:
        raise HTTPException(status_code=429, detail="Please slow down and try again.")

    request_ts = [
        float(ts)
        for ts in guard.get("request_ts", [])
        if now - float(ts) <= SESSION_WINDOW_SECONDS
    ]
    if len(request_ts) >= SESSION_REQUEST_LIMIT:
        raise HTTPException(status_code=429, detail="Session request limit reached. Please wait.")

    cost_events = [
        {"ts": float(evt.get("ts", 0)), "cost": int(evt.get("cost", 0))}
        for evt in guard.get("cost_events", [])
        if now - float(evt.get("ts", 0)) <= SESSION_COST_WINDOW_SECONDS
    ]

    used_budget = sum(evt["cost"] for evt in cost_events)
    incoming_cost = IMAGE_REQUEST_COST if is_image else TEXT_REQUEST_COST
    if used_budget + incoming_cost > SESSION_COST_BUDGET:
        raise HTTPException(status_code=429, detail="Session processing budget exceeded. Please wait.")

    request_ts.append(now)
    cost_events.append({"ts": now, "cost": incoming_cost})
    request.session["abuse_guard"] = {
        "last_ts": now,
        "request_ts": request_ts,
        "cost_events": cost_events,
    }


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str
    context: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    message: Optional[str] = None
    query: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None


class FeedbackRequest(BaseModel):
    feedback: Optional[str] = None
    rating: Optional[int] = None


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
    # Handle None or empty responses
    if not llm_text_response:
        logger.warning("LLM response is None or empty")
        return [{"title": "CineBot", "explanation": "I couldn't generate a proper response. Please try again."}]
    
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
    if not llm_parsed_recs:
        return []
    
    if not retrieved_movies:
        return llm_parsed_recs

    detailed_recs = []
    retrieved_lookup = {
        (movie.get("title") or "").lower().strip(): movie
        for movie in retrieved_movies
        if movie.get("title")
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

# NOTE: "/" route removed - React frontend handles the UI
# Backend only serves API endpoints under /api/


@app.get("/api/suggestion")
async def get_suggestion(request: Request):
    """Return a random suggestion prompt"""
    _enforce_ip_limit(request, "suggestion", SUGGESTION_RATE_LIMIT, SUGGESTION_RATE_WINDOW_SECONDS)
    return JSONResponse({"suggestion": random.choice(SUGGESTION_PROMPTS)})


@app.post("/api/feedback")
async def handle_feedback(request: Request, feedback_data: FeedbackRequest):
    """Handle user feedback"""
    _enforce_ip_limit(request, "feedback", FEEDBACK_RATE_LIMIT, FEEDBACK_RATE_WINDOW_SECONDS)
    logger.info(f"FEEDBACK RECEIVED: {feedback_data.dict()}")
    return JSONResponse({"status": "success"})


@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    from core.redis_cache import get_cache_health
    
    cache_status = get_cache_health()
    
    return JSONResponse({
        "status": "healthy",
        "cache": cache_status,
        "backend": "FastAPI",
        "version": "2.0.0"
    })


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics and performance metrics"""
    from core.redis_cache import get_genre_cache, get_vector_cache, get_graph_cache
    
    genre_cache = get_genre_cache()
    vector_cache = get_vector_cache()
    graph_cache = get_graph_cache()
    
    return JSONResponse({
        "genre_cache": {
            "size": genre_cache.size(),
            "health": genre_cache.health_check()
        },
        "vector_cache": {
            "size": vector_cache.size(),
            "health": vector_cache.health_check()
        },
        "graph_cache": {
            "size": graph_cache.size(),
            "health": graph_cache.health_check()
        }
    })


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all caches (admin endpoint)"""
    from core.redis_cache import clear_all_caches
    
    clear_all_caches()
    logger.info("All caches cleared via API")
    
    return JSONResponse({
        "status": "success",
        "message": "All caches cleared"
    })


@app.get("/api/articles")
async def get_articles(
    limit: int = 20,
    source: Optional[str] = None,
    search: Optional[str] = None
):
    """
    Get articles from movie blogs
    
    Query params:
    - limit: Maximum number of articles (default: 20)
    - source: Filter by source (e.g., 'indiewire', 'collider')
    - search: Search query for filtering articles
    """
    try:
        from external_apis.rss_feed_client import get_rss_client
        
        rss_client = get_rss_client()
        
        if search:
            articles = rss_client.search_articles(query=search, limit=limit, source=source)
        else:
            articles = rss_client.get_articles(limit=limit, source=source)
        
        return JSONResponse({
            "articles": articles,
            "count": len(articles),
            "source": source,
            "search": search
        })
        
    except Exception as e:
        logger.error(f"Error fetching articles: {e}", exc_info=True)
        return JSONResponse({
            "articles": [],
            "count": 0,
            "error": str(e)
        }, status_code=500)


@app.get("/api/articles/featured")
async def get_featured_articles(count: int = 5):
    """Get featured articles from top-tier sources"""
    try:
        from external_apis.rss_feed_client import get_rss_client
        
        rss_client = get_rss_client()
        articles = rss_client.get_featured_articles(count=count)
        
        return JSONResponse({
            "articles": articles,
            "count": len(articles)
        })
        
    except Exception as e:
        logger.error(f"Error fetching featured articles: {e}", exc_info=True)
        return JSONResponse({
            "articles": [],
            "count": 0,
            "error": str(e)
        }, status_code=500)


@app.get("/api/articles/sources")
async def get_article_sources():
    """Get list of available article sources"""
    try:
        from external_apis.rss_feed_client import get_rss_client
        
        rss_client = get_rss_client()
        sources = rss_client.get_sources()
        
        return JSONResponse({
            "sources": sources,
            "count": len(sources)
        })
        
    except Exception as e:
        logger.error(f"Error fetching sources: {e}", exc_info=True)
        return JSONResponse({
            "sources": [],
            "error": str(e)
        }, status_code=500)


@app.post("/api/chat")
async def handle_chat(
    request: Request,
    message: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    poster: Optional[UploadFile] = File(None),
    chat_history: Optional[str] = Form(None),
):
    """
    Unified chat endpoint:
    - Handles text queries (prompt box)
    - Handles poster uploads (image search)
    - Maintains a trimmed chat history in server-side session
    
    This endpoint accepts both JSON and form data
    """
    _enforce_ip_limit(request, "chat", CHAT_RATE_LIMIT, CHAT_RATE_WINDOW_SECONDS)
    _enforce_ip_limit(request, "chat-burst", CHAT_BURST_LIMIT, CHAT_BURST_WINDOW_SECONDS)

    if chat_concurrency_guard.locked():
        raise HTTPException(status_code=429, detail="Server is busy. Please retry shortly.")

    await chat_concurrency_guard.acquire()
    try:
        MAX_TURNS = 6  # how many last messages to keep in server-side history

        # Determine if we're processing JSON or form data
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # Handle JSON requests from frontend
            data = await request.json()
            user_query = data.get("message") or data.get("query")
            image_bytes = None
            
            # Get chat history from JSON and clean it
            raw_history = data.get("chat_history", [])
            history = []
            for msg in raw_history:
                # Only keep role and content, strip out context/movies
                clean_msg = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                }
                history.append(clean_msg)
        else:
            # Handle form data (for poster uploads)
            user_query = query or message
            image_bytes = None
            
            if poster:
                content = await poster.read()
                image_bytes = content if content else None
            
            # Parse chat history from form data
            if chat_history:
                try:
                    history = json.loads(chat_history)
                except json.JSONDecodeError:
                    history = []
            else:
                history = request.session.get("chat_history", [])

        # Basic validation: require either text or image
        if user_query:
            if len(user_query) > MAX_QUERY_CHARS:
                raise HTTPException(status_code=400, detail=f"Query too long. Max {MAX_QUERY_CHARS} characters.")
            history.append({"role": "user", "content": user_query})
        elif image_bytes:
            history.append({"role": "user", "content": "(Uploaded a poster)"})
        else:
            logger.error("No query or image provided in request")
            raise HTTPException(status_code=400, detail="No query or image provided.")

        _enforce_session_limits(request, is_image=bool(image_bytes))

        logger.info(f"Processing query: {user_query[:50] if user_query else 'poster upload'}")
        
        # Core RAG processing
        bot_response_text, context_movies, response_metadata = process_query(
            user_query=user_query,
            image_bytes=image_bytes,
            chat_history=history,
        )
        
        # Ensure we have a valid response
        if not bot_response_text:
            bot_response_text = "I'm having trouble processing your request. Please try rephrasing."
            logger.warning("process_query returned None for bot_response_text")
        
        if context_movies is None:
            context_movies = []

        # Add assistant message with full context (for the in-memory history / client)
        bot_message = {
            "role": "assistant",
            "content": bot_response_text,
            "context": context_movies,  # this can be heavy, so we won't store it in session
        }
        history.append(bot_message)

        # --- Trim and store history in server-side session ---
        session_history = []
        for i, msg in enumerate(history[-MAX_TURNS:]):
            stored_msg = {
                "role": msg.get("role"),
                "content": (msg.get("content") or "")[:500],
            }
            # Do NOT store full assistant `context` in the cookie-backed session.
            # Storing large `context` objects in the session causes oversized
            # Set-Cookie headers and upstream errors (Nginx: "upstream sent too big header").
            # If follow-up context is needed, store only lightweight identifiers
            # or use a server-side session store (Redis) instead.
            session_history.append(stored_msg)

        request.session["chat_history"] = session_history
        # -----------------------------------------------------------------------

        # Return clean response schema
        return JSONResponse(
            {
                "message": bot_response_text,
                "movies": context_movies,
                "response_type": response_metadata.get("response_type", "recommendation"),
                "source": response_metadata.get("source", "unknown"),
                "input_mode": response_metadata.get("input_mode", "text"),
                "meta": {
                    "has_movies": len(context_movies) > 0,
                    "count": len(context_movies)
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat API: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        print(f"\nCHAT API ERROR:\n{error_details}\n")
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred: {str(e)}"
        )
    finally:
        chat_concurrency_guard.release()


@app.get("/api/graph-data")
async def get_graph_data():
    """
    Return sample graph data for visualization
    Fetches ~100-150 nodes (movies, genres, people) with relationships
    """
    try:
        from graph_db.connection import get_driver
        
        driver = get_driver()
        
        # Query for sample movies with genres and key people
        query = """
        MATCH (m:Movie)
        WHERE m.vote_count >= 100
        WITH m ORDER BY m.vote_average DESC, m.vote_count DESC LIMIT 50
        
        OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
        OPTIONAL MATCH (m)<-[r:ACTED_IN|DIRECTED]-(p:Person)
        WHERE r.role IS NOT NULL OR r.job = 'Director'
        WITH m, collect(DISTINCT g) as genres, collect(DISTINCT p)[..5] as people
        
        RETURN m.title as movie_title, 
               m.tmdb_id as movie_id,
               m.vote_average as rating,
               [g IN genres | g.name] as genres,
               [p IN people | {name: p.name, id: p.tmdb_id}] as people
        """
        
        nodes = []
        links = []
        node_ids = set()
        
        with driver.session() as session:
            result = session.run(query)
            
            for record in result:
                movie_id = f"movie_{record['movie_id']}"
                movie_title = record['movie_title']
                rating = record['rating'] or 7.0
                
                # Add movie node
                if movie_id not in node_ids:
                    nodes.append({
                        "id": movie_id,
                        "name": movie_title,
                        "type": "movie",
                        "rating": float(rating),
                        "val": 3 + (rating / 3)  # Size based on rating
                    })
                    node_ids.add(movie_id)
                
                # Add genre nodes and links
                for genre in record['genres'] or []:
                    genre_id = f"genre_{genre.replace(' ', '_')}"
                    if genre_id not in node_ids:
                        nodes.append({
                            "id": genre_id,
                            "name": genre,
                            "type": "genre",
                            "val": 8
                        })
                        node_ids.add(genre_id)
                    
                    links.append({
                        "source": movie_id,
                        "target": genre_id,
                        "type": "HAS_GENRE"
                    })
                
                # Add person nodes and links (limited to keep graph clean)
                for person in (record['people'] or [])[:3]:  # Limit to 3 people per movie
                    person_id = f"person_{person['id']}"
                    if person_id not in node_ids:
                        nodes.append({
                            "id": person_id,
                            "name": person['name'],
                            "type": "person",
                            "val": 2
                        })
                        node_ids.add(person_id)
                    
                    links.append({
                        "source": person_id,
                        "target": movie_id,
                        "type": "WORKED_ON"
                    })
        
        logger.info(f"Graph data: {len(nodes)} nodes, {len(links)} links")
        
        return JSONResponse({
            "nodes": nodes,
            "links": links,
            "stats": {
                "total_nodes": len(nodes),
                "total_links": len(links),
                "movies": len([n for n in nodes if n['type'] == 'movie']),
                "genres": len([n for n in nodes if n['type'] == 'genre']),
                "people": len([n for n in nodes if n['type'] == 'person'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching graph data: {e}", exc_info=True)
        # Return empty graph data rather than failing
        return JSONResponse({
            "nodes": [],
            "links": [],
            "stats": {"error": str(e)}
        })


# Mount static files AFTER defining routes
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True)
