#core_rag.py

import logging
import os
import io
import threading
from typing import List, Dict, Optional, Tuple, Any
import html

import torch
from PIL import Image
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel

# --- NEW: Import LangChain components for Text-to-Cypher ---
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# ---

from config import settings
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from utils.poster_filter import is_valid_movie_poster
from core.redis_cache import get_genre_cache, get_vector_cache, get_graph_cache

# --- Global Initializations (largely the same) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from huggingface model downloads
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- OpenAI and Neo4j Clients (largely the same) ---
openai_client = None
if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
else:
    logger.error("OpenAI API Key not found in settings.")

kg = None
if hasattr(settings, 'NEO4J_URI'):
    try:
        kg = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=getattr(settings, 'NEO4J_DATABASE', "neo4j"),
            timeout=30  # Add 30 second timeout for queries
        )
        kg.refresh_schema() # Important for LangChain
        logger.info(f"Successfully connected to Neo4j and refreshed schema.")
        logger.info(f"Neo4j schema: {kg.schema}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
        kg = None
else:
    logger.error("Neo4j URI not found in settings.")
    kg = None

if kg is None:
    logger.warning("Neo4j is not available. Graph queries will not work.")


# --- CLIP Model (Lazy Loading for faster startup) ---
CLIP_MODEL_NAME_CONST = "openai/clip-vit-base-patch32"
clip_model = None
clip_processor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_loading_lock = threading.Lock()
clip_load_attempted = False

def ensure_clip_loaded():
    """Lazy-load CLIP model only when needed for poster uploads."""
    global clip_model, clip_processor, clip_load_attempted
    
    if clip_model is not None and clip_processor is not None:
        return True  # Already loaded
    
    with clip_loading_lock:
        # Double-check after acquiring lock
        if clip_model is not None and clip_processor is not None:
            return True
        
        if clip_load_attempted:
            return False  # Already tried and failed
        
        clip_load_attempted = True
        try:
            logger.info(f"⏳ Loading CLIP processor and model on {DEVICE} (lazy load)...")
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME_CONST)
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME_CONST).to(DEVICE)
            logger.info(f"CLIP model '{CLIP_MODEL_NAME_CONST}' successfully loaded.")
            return True
        except Exception as e:
            logger.exception("Failed to load CLIP model and processor.")
            return False


# --- Embedding Functions (same) ---
def get_text_embedding_openai(text_to_embed: str) -> Optional[List[float]]:
    if not openai_client: return None
    try:
        response = openai_client.embeddings.create(model="text-embedding-ada-002", input=text_to_embed)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating OpenAI embedding for text: {e}")
        return None

def get_query_image_embedding(image_bytes: bytes) -> Optional[List[float]]:
    """Compute CLIP embedding from raw image bytes without external API calls."""
    if not clip_model or not clip_processor:
        logger.error("CLIP model or processor not loaded. Cannot generate image embedding.")
        return None
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info(f"Image loaded successfully: {image.size}, mode: {image.mode}")
        
        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
            # Extract the tensor from BaseModelOutputWithPooling
            image_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            # CRITICAL: Normalize the embedding for similarity search
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Convert to list (no need for complex tensor extraction after normalization)
        embedding = image_features[0].cpu().tolist()
        logger.info(f"Successfully generated normalized image embedding of length {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to compute image embedding locally: {e}", exc_info=True)
        return None


# --- Retrieval Functions (same) ---
def retrieve_movies_by_text_similarity(query_text: str, top_k: int = 5) -> List[Dict]:
    if not kg: return []
    query_embedding = get_text_embedding_openai(query_text)
    if not query_embedding: return []
    cypher_query = """
    CALL db.index.vector.queryNodes('movie_tagline_embeddings', $top_k, $query_embedding)
    YIELD node AS m, score
    RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.tagline AS tagline,
           m.overview AS overview, m.poster_url AS poster_url, m.trailer_url AS trailer_url, score
    ORDER BY score DESC
    """
    try:
        return kg.query(cypher_query, params={"top_k": top_k, "query_embedding": query_embedding}) or []
    except Exception as e:
        logger.error(f"Error querying Neo4j for text similarity: {e}")
        return []

def retrieve_movies_by_poster_similarity(query_image_embedding: List[float], top_k: int = 5) -> List[Dict]:
    if not kg or not query_image_embedding: return []
    cypher_query = """
    CALL db.index.vector.queryNodes('movie_poster_embeddings', $top_k, $query_embedding)
    YIELD node AS m, score
    RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.tagline AS tagline,
           m.overview AS overview, m.poster_url AS poster_url, m.trailer_url AS trailer_url, score
    ORDER BY score DESC
    """
    try:
        return kg.query(cypher_query, params={"top_k": top_k, "query_embedding": query_image_embedding}) or []
    except Exception as e:
        logger.error(f"Error querying Neo4j for poster similarity: {e}")
        return []

# --- LLM Interaction & Formatting (mostly the same, with minor improvements) ---
def get_llm_response(prompt_text: str, system_message: str, max_retries: int = 2) -> str:
    if not openai_client: return "LLM client not initialized."
    
    for attempt in range(max_retries):
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.5,
                max_tokens=800
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting LLM response (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"All LLM attempts failed", exc_info=True)
                return None
            import time
            time.sleep(1)  # Brief pause before retry
    return None

def format_movies_for_llm_prompt(movies: List[Dict]) -> str:
    if not movies: return "No relevant movies were found."
    context_parts = []
    for i, movie in enumerate(movies[:5]):
        title = html.escape(movie.get('title', 'N/A'))
        tagline = html.escape(movie.get('tagline', ''))
        overview = html.escape(movie.get('overview', ''))[:250]
        # NEW: Adding an index for easier reference in follow-ups
        movie_str = f"--- Movie Index {i+1}: {title} ---\nTagline: {tagline}\nOverview: {overview}\n---"
        context_parts.append(movie_str)
    return "\n\n".join(context_parts)

def format_chat_history_for_llm(chat_history: List[Dict]) -> str:
    if not chat_history: return ""
    history_str = "This is the conversation history:\n"
    for message in chat_history:
        role = "User" if message["role"] == "user" else "CineBot"
        content = message.get("content", "")
        # NEW: We won't show the raw context to the LLM in the history
        if "context" in message:
            content += " (CineBot provided some recommendations)"
        history_str += f"{role}: {content}\n"
    return history_str

# --- Intent Classification --
def classify_query_intent(user_query: str, chat_history: List[Dict]) -> str:
    if not openai_client: return "vector_search"
    
    # Quick heuristic checks for common follow-up patterns
    import re
    query_lower = user_query.lower().strip()
    
    # PRIORITY 0: Check for pronouns referring to previous context - ALWAYS follow_up
    # These must be checked BEFORE factual patterns because pronouns indicate context dependency
    pronoun_patterns = [
        r'\b(him|her|them|his|hers|their|he|she|they)\b',
        r'\b(it|this|that|these|those)\b',
        r'\b(first|second|third|last|other)\s+(one|movie|film)',
        r'^(and|but|also|or)\b',
    ]
    
    if chat_history and len(chat_history) >= 1:
        for pattern in pronoun_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"[IntentClassifier] Detected pronoun/reference pattern: {pattern} → follow_up")
                return "follow_up"
        
        # PRIORITY 0.5: Check if query mentions a movie title from previous context
        # This must come BEFORE factual patterns to handle "who directed [movie from context]"
        # Find the last assistant message (not the current user message)
        last_bot_message = None
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                last_bot_message = msg
                break
        
        if last_bot_message:
            previous_context_movies = last_bot_message.get("context", [])
            logger.info(f"[IntentClassifier] Checking previous context: {len(previous_context_movies)} movies found")
            if previous_context_movies:
                logger.info(f"[IntentClassifier] First movie title: {previous_context_movies[0].get('title', 'N/A')}")
            
            if previous_context_movies:
                import unicodedata
                def normalize_title(title):
                    # Remove accents
                    title = unicodedata.normalize('NFD', title)
                    title = ''.join(c for c in title if unicodedata.category(c) != 'Mn')
                    # Remove special characters and lowercase
                    title = re.sub(r'[^\w\s]', '', title).lower().strip()
                    return title
                
                normalized_query = normalize_title(user_query)
                logger.info(f"[IntentClassifier] Normalized query: '{normalized_query}'")
                
                for movie in previous_context_movies:
                    movie_title = movie.get("title", "")
                    if not movie_title or len(movie_title) <= 3:
                        continue
                    
                    normalized_movie_title = normalize_title(movie_title)
                    logger.info(f"[IntentClassifier] Checking movie: '{movie_title}' → normalized: '{normalized_movie_title}'")
                    
                    if normalized_movie_title and normalized_movie_title in normalized_query:
                        logger.info(f"[IntentClassifier] Query mentions previous movie '{movie_title}' → follow_up")
                        return "follow_up"
    
    # PRIORITY 1: Factual questions - ALWAYS graph_search
    # These patterns indicate specific factual queries that should use the graph database
    factual_question_patterns = [
        r'\b(who|what)\s+(directed|made|created|wrote|produced)\b',
        r'\b(director|actor|cast|crew)\s+of\b',
        r'\b(who|what)\s+(starred?|acted?)\s+(in|on)\b',
        r'\b(what|which)\s+movies?\s+(did|by|from|with)\b',
        r'\b(list|show|give|tell)\s+.*\s+(movies?|films?)\s+(by|from|with|starring)\b',
        r'\bfilms?\s+(directed|made|by)\b',
    ]
    
    for pattern in factual_question_patterns:
        if re.search(pattern, query_lower):
            logger.info(f"[IntentClassifier] Detected factual question pattern: {pattern} → graph_search")
            return "graph_search"
    
    # Pattern 1: Other follow-up patterns
    follow_up_patterns = [
        r'\btell me (more|about)\s+(it|this|that|the\s+(first|second|third|last))',
        r'\bwhat about\b',
        r'\bhow about\b',
        r'\bwhich one\b',
        r'\bany (other|more)\b',
        r'\b(when|where|who|what)\s+(was|were|is|are)\s+(it|this|that|they)',
    ]
    
    # If there's chat history and the query matches follow-up patterns
    if chat_history and len(chat_history) >= 1:
        for pattern in follow_up_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"[IntentClassifier] Detected follow-up pattern: {pattern}")
                return "follow_up"
    
    # Pattern 2: Check for specific named entities (directors, actors, specific movies)
    # These should be graph_search  
    # Extract potential names from the query
    has_specific_entity = bool(re.search(
        r'\b(nolan|christopher nolan|spielberg|steven spielberg|tarantino|quentin tarantino|'
        r'scorsese|martin scorsese|kubrick|stanley kubrick|hitchcock|alfred hitchcock|'
        r'coppola|francis ford coppola|fincher|david fincher|anderson|wes anderson|'
        r'villeneuve|denis villeneuve|cameron|james cameron|'
        r'cruise|tom cruise|hanks|tom hanks|dicaprio|leonardo dicaprio|pitt|brad pitt|'
        r'depp|johnny depp|downey|robert downey|freeman|morgan freeman|'
        r'washington|denzel washington|johansson|scarlett johansson|'
        r'lawrence|jennifer lawrence|streep|meryl streep|'
        r'matrix|inception|godfather|interstellar|titanic|pulp fiction|shawshank|'
        r'forrest gump|avatar|dark knight|fight club|memento)\\b',
        query_lower
    ))
    
    # Pattern 3: Vague descriptors (genres, moods, themes) without specific names
    # These should be vector_search
    has_vague_descriptor = bool(re.search(
        r'\b(romantic|comedy|action|thriller|horror|sci-fi|drama|adventure|fantasy|mystery|'
        r'funny|sad|scary|exciting|suspenseful|emotional|heartwarming|dark|intense|light|beautiful|love|'
        r'feel-good|uplifting|depressing|mind-bending|classic|recent|old|new|popular)\b',
        query_lower
    )) and not has_specific_entity
    
    # If query has vague descriptors, it's vector_search
    if has_vague_descriptor:
        logger.info(f"[IntentClassifier] Detected vague descriptors → vector_search")
        return "vector_search"
    
    # If query has specific entities, it's graph_search
    if has_specific_entity:
        logger.info(f"[IntentClassifier] Detected specific entity → graph_search")
        return "graph_search"
    
    # Heuristic: If the last assistant message had no context, treat next as graph_search
    if chat_history and chat_history[-1]["role"] == "assistant" and not chat_history[-1].get("context"):
       logger.info("[IntentClassifier] Last assistant message had no context → Forcing intent: graph_search")
       return "graph_search"

    history_context = (
        "This is the first message." if not chat_history
        else f"The last message from the bot was: '{chat_history[-1]['content'][:200]}...'"
    )

    prompt = f"""
You are an intent classifier for a movie chatbot.

Your job is to classify the user's latest message into ONE of exactly these categories:

1. `follow_up`: Questions about previously recommended movies or continuing the conversation
   Examples: "Tell me more about it", "What's the rating?", "When was it released?", "Who starred in the first one?", "Is it scary?", "Which one is newest?", "Tell me about the second movie", "Who directed Miami Vice?" (if Miami Vice was previously discussed)

2. `graph_search`: Factual questions using SPECIFIC names (directors, actors, movie titles) NOT from previous context
   Examples: "Who directed Interstellar?", "List movies with Tom Cruise", "Show me Nolan films", "Movies by Spielberg", "Tell me about The Matrix"

3. `vector_search`: Vague recommendations based on mood, theme, genres, or descriptions WITHOUT specific names
   Examples: "Give me something romantic", "I want an action movie", "Recommend thriller films", "Tell me about romantic movies", "Show me comedy films", "beautiful love movie"

Rules:
- If the query references previous results (it, this, that, the first one, etc.) OR mentions a movie from previous results → `follow_up`
- If the query has SPECIFIC names (directors, actors, movie titles) NOT from context → `graph_search`
- If the query has VAGUE descriptors (genres, moods, themes, adjectives) without specific names → `vector_search`
- Always return just one of: graph_search, vector_search, follow_up

Conversation context:
{history_context}

User's latest query: "{html.escape(user_query)}"

Return only one word:"""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        content = completion.choices[0].message.content or ""
        intent_raw = content.strip().lower().replace(" ", "_")
        intent = intent_raw.splitlines()[0] if intent_raw else "vector_search"
        logger.info(f"[IntentClassifier] Query: '{user_query}' → Classified as: {intent}")

        if intent in {"graph_search", "vector_search", "follow_up"}:
            return intent

        logger.warning(f"Unclear intent '{intent}', defaulting to vector_search.")
        return "vector_search"
    except Exception as e:
        logger.error(f"Error classifying user intent: {e}")
        return "vector_search"


def handle_vector_search(user_query: str, chat_history: List[Dict], top_k: int = 10, num_rec: int = 5) -> Tuple[str, List[Dict]]:
    """Handles semantic search using vector similarity or genre-based search."""
    logger.info("Handling query with VECTOR SEARCH")
    
    import re
    
    # First, try to detect if this is a genre-based query
    genre_mapping = {
        'romantic': 'Romance',
        'romance': 'Romance',
        'love': 'Romance',
        'comedy': 'Comedy',
        'funny': 'Comedy',
        'action': 'Action',
        'adventure': 'Adventure',
        'thriller': 'Thriller',
        'horror': 'Horror',
        'scary': 'Horror',
        'drama': 'Drama',
        'sci-fi': 'Science Fiction',
        'scifi': 'Science Fiction',
        'science fiction': 'Science Fiction',
        'fantasy': 'Fantasy',
        'animation': 'Animation',
        'animated': 'Animation',
        'documentary': 'Documentary',
        'crime': 'Crime',
        'mystery': 'Mystery',
        'war': 'War',
        'western': 'Western',
        'musical': 'Music',
        'family': 'Family'
    }
    
    # Check if query mentions a genre
    detected_genre = None
    query_lower = user_query.lower()
    for keyword, genre_name in genre_mapping.items():
        if keyword in query_lower:
            detected_genre = genre_name
            logger.info(f"Detected genre '{genre_name}' from query")
            break
    
    # If genre detected, use graph-based genre search
    if detected_genre and kg:
        logger.info(f"Using genre-based search for: {detected_genre}")
        
        # Check cache first
        genre_cache = get_genre_cache()
        cache_key = f"genre_{detected_genre}_top15"
        cached_movies = genre_cache.get(cache_key, params={"genre": detected_genre})
        
        if cached_movies is not None:
            logger.info(f"Cache HIT: Using cached results for genre '{detected_genre}'")
            genre_movies = cached_movies
        else:
            # Optimized query with index hint
            cypher = """
            MATCH (g:Genre {name: $genre_name})<-[:HAS_GENRE]-(m:Movie)
            WHERE m.vote_average IS NOT NULL AND m.popularity IS NOT NULL
            RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.tagline AS tagline, 
                   m.overview AS overview, m.poster_url AS poster_url, 
                   m.trailer_url AS trailer_url, m.vote_average AS vote_average,
                   m.popularity AS popularity
            ORDER BY m.vote_average DESC, m.popularity DESC
            LIMIT 15
            """
            try:
                logger.info(f"⏳ Executing genre query for '{detected_genre}'...")
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Query execution exceeded timeout")
                
                # Set up timeout alarm (Unix only)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(15)  # 15 second timeout for genre query
                    genre_movies = kg.query(cypher, params={"genre_name": detected_genre}) or []
                    signal.alarm(0)  # Cancel the alarm
                    
                    # Cache the results for future queries
                    if genre_movies:
                        genre_cache.set(cache_key, genre_movies, params={"genre": detected_genre})
                        logger.info(f"💾 Cached {len(genre_movies)} movies for genre '{detected_genre}'")
                except (AttributeError, ValueError):
                    # Windows doesn't support SIGALRM, fallback to no timeout
                    genre_movies = kg.query(cypher, params={"genre_name": detected_genre}) or []
                    if genre_movies:
                        genre_cache.set(cache_key, genre_movies, params={"genre": detected_genre})
            except TimeoutError as te:
                logger.error(f"Genre query timed out after 15s: {te}. Falling back to vector search.")
                genre_movies = []
            except Exception as e:
                logger.error(f"Genre-based search failed: {e}. Falling back to vector search.")
                genre_movies = []
        
        if genre_movies:
            logger.info(f"Found {len(genre_movies)} movies with genre '{detected_genre}'")
            movie_context = format_movies_for_llm_prompt(genre_movies)
            history_context = format_chat_history_for_llm(chat_history)
            system_message = "You are CineBot, a movie recommender. Select the best movies from the context provided and explain why in 1-2 sentences. Format your response exactly as: MOVIE: [Title]\nEXPLANATION: [Your text]"
            prompt = f"""{history_context}\nBased on the user's request for '{html.escape(user_query)}', I have found the following {detected_genre} movies:\nCONTEXT:\n{movie_context}\n\nTASK: Select the {num_rec} best movies from the CONTEXT. For EACH, respond in the required format.\n"""
            llm_response = get_llm_response(prompt, system_message)
            if llm_response:
                return llm_response, genre_movies
        else:
            logger.warning(f"No movies found for genre '{detected_genre}', falling back to vector search")
    
    # Fallback to vector search
    logger.info("Using vector similarity search")
    query_variations = [user_query]
    
    # Add genre-specific variations for vector search
    if re.search(r'\b(romantic|romance|love)\b', user_query, re.I):
        query_variations.extend(["romantic love story", "heartwarming romance", "beautiful love movie"])
    if re.search(r'\b(comedy|funny|humor)\b', user_query, re.I):
        query_variations.extend(["funny comedy", "hilarious movie"])
    if re.search(r'\b(action|adventure)\b', user_query, re.I):
        query_variations.extend(["action-packed adventure", "thrilling action"])
    
    # Try each variation until we get results
    retrieved_movies = []
    for query_var in query_variations:
        retrieved_movies = retrieve_movies_by_text_similarity(query_var, top_k=top_k)
        if retrieved_movies:
            logger.info(f"Found {len(retrieved_movies)} movies using query variation: {query_var}")
            break
    
    # If no vector results, return helpful message
    if not retrieved_movies:
        return "I apologize, but I couldn't find movies matching that description in our current database. This is a demo version with limited movie data. Could you try asking about popular genres like action, romance, or comedy?", []

    movie_context = format_movies_for_llm_prompt(retrieved_movies)
    history_context = format_chat_history_for_llm(chat_history)

    system_message = "You are CineBot, a movie recommender. Select the best movies from the context provided and explain why in 1-2 sentences. Format your response exactly as: MOVIE: [Title]\nEXPLANATION: [Your text]"
    prompt = f"""{history_context}
Based on the user's request for "{html.escape(user_query)}", I have found the following movies:
CONTEXT:
{movie_context}

TASK: Select the {num_rec} best movies from the CONTEXT. For EACH, respond in the required format.
"""
    llm_response = get_llm_response(prompt, system_message)
    
    # Fallback if LLM fails: create basic recommendations from retrieved movies
    if not llm_response:
        logger.warning("LLM failed, using fallback movie presentation")
        fallback_response = ""
        for i, movie in enumerate(retrieved_movies[:num_rec]):
            title = movie.get('title', 'Unknown')
            overview = movie.get('overview', 'No description available')[:150]
            fallback_response += f"MOVIE: {title}\nEXPLANATION: {overview}...\n\n"
        return fallback_response.strip(), retrieved_movies
    
    return llm_response, retrieved_movies


def handle_graph_search(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
    """Handles factual questions using Text-to-Cypher with a custom, few-shot prompt."""
    logger.info("Handling query with GRAPH SEARCH (Text-to-Cypher)")
    if not kg or not settings.OPENAI_API_KEY:
        return "My graph search functionality is not configured.", []

    CYPHER_GENERATION_TEMPLATE = """
    Task: Generate Cypher statement to query a graph database.
    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.
    Do not return any explanations or apologies.
    Your response must be in a markdown code block with the "cypher" tag.
    For person names, use CONTAINS for partial matching (e.g., 'Nolan' should match 'Christopher Nolan').
    For movie titles, use exact matching OR CONTAINS if not sure.
    Always return movie details: m.title, m.overview, m.poster_url, m.trailer_url, m.tmdb_id
    
    Schema:
    {schema}

    ---
    Here are some examples of questions and their corresponding Cypher queries.

    Question: Who directed the movie The Matrix?
    Cypher:
    ```cypher
    MATCH (p:Person)-[:DIRECTED]->(m:Movie {{title: 'The Matrix'}})
    RETURN p.name AS director, m.title AS movie
    ```

    Question: Who directed Titanic?
    Cypher:
    ```cypher
    MATCH (p:Person)-[:DIRECTED]->(m:Movie)
    WHERE m.title CONTAINS 'Titanic'
    RETURN p.name AS director, m.title AS movie
    ```

    Question: What movies did Tom Hanks act in?
    Cypher:
    ```cypher
    MATCH (p:Person {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie)
    RETURN m.title AS title, m.overview AS overview, m.poster_url AS poster_url, m.trailer_url AS trailer_url, m.tmdb_id AS tmdb_id
    LIMIT 10
    ```

    Question: Tell me a movie from Nolan
    Cypher:
    ```cypher
    MATCH (p:Person)-[:DIRECTED]->(m:Movie)
    WHERE p.name CONTAINS 'Nolan'
    RETURN m.title AS title, m.overview AS overview, m.poster_url AS poster_url, m.trailer_url AS trailer_url, m.tmdb_id AS tmdb_id
    LIMIT 10
    ```

    Question: Show me films by Spielberg
    Cypher:
    ```cypher
    MATCH (p:Person)-[:DIRECTED]->(m:Movie)
    WHERE p.name CONTAINS 'Spielberg'
    RETURN m.title AS title, m.overview AS overview, m.poster_url AS poster_url, m.trailer_url AS trailer_url, m.tmdb_id AS tmdb_id
    LIMIT 10
    ```
    
    Question: Give me five movies from Nolan
    Cypher:
    ```cypher
    MATCH (p:Person)-[:DIRECTED]->(m:Movie)
    WHERE p.name CONTAINS 'Nolan'
    RETURN m.title AS title, m.overview AS overview, m.poster_url AS poster_url, m.trailer_url AS trailer_url, m.tmdb_id AS tmdb_id
    LIMIT 5
    ```
    ---

    Now, generate the Cypher statement for this question:
    Question: {question}
    """

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE
    )

    try:
        # First, generate and execute the Cypher query to get movie data
        cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
            qa_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
            graph=kg,
            verbose=True,
            cypher_prompt=cypher_prompt,
            allow_dangerous_requests=True,
            return_intermediate_steps=True
        )
        result = cypher_chain.invoke({"query": user_query})
        
        # Extract the intermediate steps to get the actual Cypher results
        intermediate_steps = result.get("intermediate_steps", [])
        movies_list = []
        
        if intermediate_steps:
            # The Cypher query result is in intermediate_steps
            cypher_result = intermediate_steps[0].get("context", [])
            
            # Convert Cypher results to our standard movie format
            for item in cypher_result:
                if isinstance(item, dict):
                    movie = {
                        "title": item.get("m.title") or item.get("title"),
                        "overview": item.get("m.overview") or item.get("overview"),
                        "poster_url": item.get("m.poster_url") or item.get("poster_url"),
                        "trailer_url": item.get("m.trailer_url") or item.get("trailer_url"),
                        "tmdb_id": item.get("m.tmdb_id") or item.get("tmdb_id"),
                    }
                    # Only add if it has at least a title
                    if movie["title"]:
                        movies_list.append(movie)
        
        answer = result.get("result", "I apologize, but I couldn't find that information in our current database. This is a demo version with limited movie data. Feel free to ask about other popular movies or directors!")
        
        # If we got movies, format a nice response
        if movies_list:
            movie_context = format_movies_for_llm_prompt(movies_list)
            system_message = "You are CineBot. Present the movies from the context with brief explanations. Format your response exactly as: MOVIE: [Title]\nEXPLANATION: [Your text]"
            prompt = f"""The user asked: "{html.escape(user_query)}"
I found these movies:
CONTEXT:
{movie_context}

TASK: Present up to 5 movies from the CONTEXT with brief explanations in the required format.
"""
            llm_response = get_llm_response(prompt, system_message)
            return llm_response, movies_list
        
        # If no movies but we have a direct answer, return it
        if answer and "couldn't find" not in answer.lower():
            return answer, []
        
    except Exception as e:
        logger.error(f"Error during GraphCypherQAChain execution: {e}", exc_info=True)
    
    # Fallback: Try direct pattern-based Cypher queries
    logger.info("Attempting fallback pattern-based query")
    import re
    
    # Pattern 1: "who directed [movie]" or "director of [movie]"
    director_match = re.search(r'(?:who directed|director of|directed by whom)\s+(?:the\s+)?(?:movie\s+)?["\']?([\w\s]+?)["\']?(?:\?|$)', user_query, re.I)
    if director_match:
        movie_title = director_match.group(1).strip()
        logger.info(f"Attempting director query for movie: {movie_title}")
        
        # Try case-insensitive search
        cypher = """
        MATCH (p:Person)-[:DIRECTED]->(m:Movie)
        WHERE toLower(m.title) CONTAINS toLower($title) OR toLower(m.title) = toLower($exact_title)
        RETURN p.name AS director, m.title AS movie
        LIMIT 5
        """
        try:
            results = kg.query(cypher, params={"title": movie_title, "exact_title": movie_title})
            if results:
                if len(results) == 1:
                    return f"{results[0]['movie']} was directed by {results[0]['director']}.", []
                else:
                    directors_info = ", ".join([f"{r['movie']} by {r['director']}" for r in results])
                    return f"I found these: {directors_info}", []
        except Exception as e:
            logger.error(f"Fallback director query failed: {e}")
    
    # Pattern 2: "movies by [director]" or "[director] films"
    director_name_match = re.search(r'(?:movies? by|films? by|directed by|from)\s+([A-Z][\w\s]+?)(?:\?|$|\s+movie)', user_query, re.I)
    if director_name_match:
        director_name = director_name_match.group(1).strip()
        cypher = """
        MATCH (p:Person)-[:DIRECTED]->(m:Movie)
        WHERE p.name CONTAINS $name
        RETURN m.title AS title, m.overview AS overview, m.poster_url AS poster_url, 
               m.trailer_url AS trailer_url, m.tmdb_id AS tmdb_id, p.name AS director
        LIMIT 10
        """
        try:
            results = kg.query(cypher, params={"name": director_name})
            if results:
                movies_list = [{"title": r["title"], "overview": r["overview"], 
                               "poster_url": r["poster_url"], "trailer_url": r["trailer_url"],
                               "tmdb_id": r["tmdb_id"]} for r in results]
                movie_context = format_movies_for_llm_prompt(movies_list)
                system_message = "You are CineBot. Present the movies with brief explanations. Format your response exactly as: MOVIE: [Title]\nEXPLANATION: [Your text]"
                prompt = f"""The user asked for movies by {results[0]['director']}:
CONTEXT:
{movie_context}

TASK: Present up to 5 movies in the required format.
"""
                llm_response = get_llm_response(prompt, system_message)
                if llm_response:
                    return llm_response, movies_list
        except Exception as e:
            logger.error(f"Fallback director movies query failed: {e}")
    
    # Final fallback: try vector search
    logger.info("All graph search attempts failed, falling back to vector search")
    return handle_vector_search(user_query, chat_history)


# def handle_follow_up(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
#     """Handles follow-up questions based on the last interaction's context."""
#     logger.info("Handling query as a FOLLOW-UP")
#     if not chat_history:
#         return "There is no previous conversation to follow up on.", []

#     last_bot_message = chat_history[-1]
#     previous_context_movies = last_bot_message.get("context", [])

#     if not previous_context_movies:
#         return "I don't have a specific list of movies from our last chat to discuss.", []
    
#     movie_context = format_movies_for_llm_prompt(previous_context_movies)
#     history_context = format_chat_history_for_llm(chat_history[:-1])

#     system_message = "You are CineBot. Answer the user's follow-up question based *only* on the previous list of movies provided in the CONTEXT. Be concise."
#     prompt = f"""{history_context}
# I previously recommended the following movies:
# CONTEXT:
# {movie_context}

# Now, the user has a follow-up question: "{html.escape(user_query)}"

# TASK: Answer the user's question based on the information in the CONTEXT. If you cannot answer from the context, say so.
# """
#     answer = get_llm_response(prompt, system_message)
#     return answer, previous_context_movies
def handle_follow_up(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
    logger.info("="*70)
    logger.info("FOLLOW-UP HANDLER ACTIVATED")
    logger.info(f"   Query: '{user_query}'")
    logger.info("="*70)

    if not chat_history:
        return "I apologize, but there's no previous conversation to reference. This is a demo version with limited context. Feel free to start a new movie query!", []

    # Find the last assistant message (not the current user message)
    last_bot_message = None
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            last_bot_message = msg
            break
    
    if not last_bot_message:
        logger.warning("No previous assistant message found in chat history")
        return "I apologize, but I don't have any previous context to reference. Please ask a new question!", []
    
    previous_context_movies = last_bot_message.get("context", [])
    previous_content = last_bot_message.get("content", "")
    logger.info(f"Previous context has {len(previous_context_movies)} movies")
    if previous_context_movies:
        logger.info(f"   Movies: {[m.get('title', 'Unknown') for m in previous_context_movies[:3]]}")
    
    import re
    
    # Check if user is asking about movies from a person using pronouns (him, her, them)
    # e.g., "tell me some more movie from him" after "Christopher Nolan directed The Dark Knight"
    pronoun_movie_pattern = re.search(r'(?:give me|show me|tell me|recommend|suggest|list|find).*(?:movies?|films?).*\b(from|by)\s+(him|her|them|he|she|they)\b', user_query, re.I)
    if pronoun_movie_pattern:
        # Try to extract a person's name from the previous answer
        # Look for patterns like "directed by X" or "X directed" or standalone capitalized names
        person_name_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) directed',  # "Luc Besson directed" or "Nolan directed"
            r'directed by ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "directed by Luc Besson"
            r'(?:actor|actress|director|filmmaker)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Names at start of sentence
        ]
        
        director_name = None
        for pattern in person_name_patterns:
            match = re.search(pattern, previous_content)
            if match:
                director_name = match.group(1).strip()
                logger.info(f"Extracted director/person name from previous context: {director_name}")
                break
        
        if director_name:
            # Construct a new query with the actual name and route to graph_search
            new_query = user_query.replace(pronoun_movie_pattern.group(2), director_name)
            logger.info(f"Pronoun resolved query: '{user_query}' → '{new_query}'")
            return handle_graph_search(new_query, chat_history)
        else:
            # If we can't extract a name but detected pronouns, explain the issue
            logger.warning(f"Could not extract person name from previous content: '{previous_content[:100]}'")
            return "I apologize, but I couldn't determine who you're referring to from our previous conversation. Could you specify the director or actor name?", []
    
    # Check if user is asking for more movies from a director mentioned in previous answer
    # e.g., "can you give me five movies from Nolan" after an answer mentioning Nolan
    more_movies_pattern = re.search(r'(?:give me|show me|tell me about|recommend|list|find)\s+(?:\w+\s+)?movies?\s+(?:from|by)\s+(\w+)', user_query, re.I)
    if more_movies_pattern and not previous_context_movies:
        director_mention = more_movies_pattern.group(1)
        # Check if this director was mentioned in the previous answer
        if director_mention.lower() in previous_content.lower():
            logger.info(f"User asking for movies by {director_mention} mentioned in previous answer")
            return handle_graph_search(user_query, chat_history)

    # 🚨 Reroute early if no real movie data
    if not previous_context_movies or not any("title" in m for m in previous_context_movies):
        logger.info("No movie context or invalid context found. Rerouting to graph_search.")
        return handle_graph_search(user_query, chat_history)

    # --- Pronoun resolution for follow-up questions ---
    resolved_query = user_query
    import re
    
    # Check if query mentions a specific movie title from context
    selected_movie_idx = None
    for idx, movie in enumerate(previous_context_movies):
        movie_title = movie.get("title", "")
        if not movie_title:
            continue
        
        # Use normalized matching for better detection
        import unicodedata
        def normalize_for_matching(text):
            # Remove accents
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
            # Remove special characters and lowercase
            text = re.sub(r'[^\w\s]', '', text).lower().strip()
            return text
        
        normalized_movie_title = normalize_for_matching(movie_title)
        normalized_query = normalize_for_matching(user_query)
        
        if normalized_movie_title and normalized_movie_title in normalized_query:
            selected_movie_idx = idx
            logger.info(f"Found specific movie reference: {movie.get('title')} (index {idx}) via normalized matching")
            break
    
    # If no specific movie found, check for pronouns
    if selected_movie_idx is None:
        pronoun_patterns = [
            (r"\b(it|this|that)\b", 0),
            (r"\bthe first (one|movie|film)\b", 0),
            (r"\bthe second (one|movie|film)\b", 1),
            (r"\bthe third (one|movie|film)\b", 2),
            (r"\bthe (last|final) (one|movie|film)\b", -1)
        ]
        
        for pattern, idx in pronoun_patterns:
            if re.search(pattern, user_query, re.I):
                selected_movie_idx = idx
                try:
                    if idx == -1:
                        movie = previous_context_movies[-1]
                    else:
                        movie = previous_context_movies[idx]
                    title = movie.get("title")
                    if title:
                        resolved_query = re.sub(pattern, f"'{title}'", user_query, flags=re.I)
                        logger.info(f"Resolved pronoun to movie title '{title}' (index {idx})")
                except Exception as e:
                    logger.warning(f"Failed to resolve pronoun '{pattern}': {e}")
                break

    # --- Check if query needs additional info from graph database ---
    needs_graph_data = any([
        re.search(r'\b(director|directed|filmmaker)\b', user_query, re.I),
        re.search(r'\b(actor|actress|star|cast|acted)\b', user_query, re.I),
        re.search(r'\b(release|released|came out|year)\b', user_query, re.I),
        re.search(r'\b(rating|score|rated)\b', user_query, re.I),
    ])

    if needs_graph_data and selected_movie_idx is not None:
        # Fetch additional data from Neo4j for the specific movie
        try:
            if selected_movie_idx == -1:
                selected_movie = previous_context_movies[-1]
            else:
                selected_movie = previous_context_movies[selected_movie_idx]
            
            movie_title = selected_movie.get("title")
            movie_tmdb_id = selected_movie.get("tmdb_id")
            
            if kg:
                logger.info(f"Fetching additional graph data for: {movie_title} (tmdb_id: {movie_tmdb_id})")
                
                # Try exact match by tmdb_id first (most reliable)
                if movie_tmdb_id:
                    cypher_query = """
                    MATCH (m:Movie {tmdb_id: $tmdb_id})
                    OPTIONAL MATCH (d:Person)-[:DIRECTED]->(m)
                    OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)
                    RETURN m.title AS title, 
                           m.release_date AS release_date,
                           m.vote_average AS rating,
                           m.overview AS overview,
                           COLLECT(DISTINCT d.name) AS directors,
                           COLLECT(DISTINCT a.name)[0..5] AS actors
                    LIMIT 1
                    """
                    result = kg.query(cypher_query, params={"tmdb_id": str(movie_tmdb_id)})
                # Fallback: flexible title matching (case-insensitive, handles variations)
                elif movie_title:
                    cypher_query = """
                    MATCH (m:Movie)
                    WHERE toLower(m.title) CONTAINS toLower($title) 
                       OR toLower(m.title) = toLower($exact_title)
                    OPTIONAL MATCH (d:Person)-[:DIRECTED]->(m)
                    OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)
                    RETURN m.title AS title, 
                           m.release_date AS release_date,
                           m.vote_average AS rating,
                           m.overview AS overview,
                           COLLECT(DISTINCT d.name) AS directors,
                           COLLECT(DISTINCT a.name)[0..5] AS actors
                    LIMIT 1
                    """
                    result = kg.query(cypher_query, params={"title": movie_title, "exact_title": movie_title})
                else:
                    result = None
                
                if result and len(result) > 0:
                    movie_data = result[0]
                    # Filter out None and empty strings from lists
                    directors = [d for d in movie_data.get('directors', []) if d]
                    actors = [a for a in movie_data.get('actors', []) if a]
                    release_date = movie_data.get('release_date', 'N/A')
                    rating = movie_data.get('rating', 'N/A')
                    logger.info(f"Graph data received:")
                    logger.info(f"   Title: {movie_data.get('title')}")
                    logger.info(f"   Directors: {directors}")
                    logger.info(f"   Actors: {actors[:3]}")
                    logger.info(f"   Release: {release_date}, Rating: {rating}")
                    
                    # If user is asking about director but we have no director info, reroute to graph_search
                    if re.search(r'\b(director|directed)\b', user_query, re.I) and not directors:
                        logger.info(f"No director info found for {movie_title}, rerouting to graph_search")
                        return handle_graph_search(user_query, chat_history)
                    
                    # Enhance the context with graph data
                    enriched_context = f"""
Movie: {movie_data.get('title')}
Release Date: {release_date}
Rating: {rating}/10
Directors: {', '.join(directors) if directors else 'N/A'}
Main Cast: {', '.join(actors[:3]) if actors else 'N/A'}
Overview: {movie_data.get('overview', 'N/A')}
"""
                    
                    system_message = "You are CineBot. Answer the user's question based on the movie information provided. Be concise and friendly."
                    prompt = f"""The user asked about a movie: "{html.escape(user_query)}"

Here's the detailed information:
{enriched_context}

TASK: Answer the user's question based on this information.
"""
                    answer = get_llm_response(prompt, system_message)
                    logger.info(f"LLM generated answer: '{answer[:100]}...'") if answer else logger.warning("LLM returned empty answer")
                    
                    # Fallback if LLM fails: provide direct answer based on query type
                    if not answer:
                        if re.search(r'\b(director|directed)\b', user_query, re.I):
                            answer = f"{movie_data.get('title')} was directed by {', '.join(directors) if directors else 'unknown'}."
                        elif re.search(r'\b(actor|actress|star|cast)\b', user_query, re.I):
                            answer = f"{movie_data.get('title')} stars {', '.join(actors[:3]) if actors else 'unknown'}."
                        elif re.search(r'\b(release|released|year)\b', user_query, re.I):
                            answer = f"{movie_data.get('title')} was released on {release_date}."
                        elif re.search(r'\b(rating|score)\b', user_query, re.I):
                            answer = f"{movie_data.get('title')} has a rating of {rating}/10."
                        else:
                            answer = enriched_context.strip()
                    
                    logger.info(f"Returning answer with {len(previous_context_movies)} context movies")
                    return answer, previous_context_movies
                else:
                    # No results found in graph query, reroute to graph_search if asking about specific details
                    logger.warning(f"No graph data found in Neo4j")
                    if re.search(r'\b(director|directed|actor|actress|star|cast|release|year)\b', user_query, re.I):
                        logger.info(f"   Rerouting to graph_search...")
                        return handle_graph_search(user_query, chat_history)
        except Exception as e:
            logger.error(f"Error fetching graph data for follow-up: {e}")
            # Continue with regular follow-up handling

    # Standard LLM-based follow-up on movie list
    movie_context = format_movies_for_llm_prompt(previous_context_movies)
    history_context = format_chat_history_for_llm(chat_history[-3:])  # Last 3 messages for context

    system_message = (
        "You are CineBot, a helpful movie assistant. Answer the user's follow-up question based on the movies "
        "from the previous conversation. Be conversational and helpful. If you need information not in the context, "
        "mention that you can search for more details if needed."
    )

    prompt = f"""{history_context}
I previously recommended these movies:
CONTEXT:
{movie_context}

Now, the user asks: "{html.escape(resolved_query)}"

TASK: Answer the user's question based on the CONTEXT. Be conversational and helpful.
"""

    answer = get_llm_response(prompt, system_message)

    # Handle None or failed responses
    if not answer or (
        "cannot answer" in answer.lower()
        or "don't have" in answer.lower()
        or "not in the context" in answer.lower()
    ):
        logger.info("LLM failed to answer follow-up. Fallback to graph search.")
        return handle_graph_search(resolved_query, chat_history)

    return answer, previous_context_movies


def recommend_by_poster_image(image_bytes: bytes, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
    """Generates recommendations from a poster image."""
    logger.info("Handling query with IMAGE SEARCH")
    
    # Lazy-load CLIP model if not already loaded
    if not ensure_clip_loaded():
        logger.error("CLIP model not available for poster search")
        return (
            "I apologize, but the image search feature is currently unavailable. "
            "This is a demo version with limited resources. Please try text-based queries!",
            []
        )
    
    # Poster validation DISABLED for performance (was loading duplicate CLIP model)
    # The is_valid_movie_poster() function loads a second CLIP model via transformers pipeline
    # which is very slow. We'll rely on the embedding similarity check instead.
    # If users upload non-posters, they'll simply get irrelevant results.
    
    # Generate embedding
    query_embedding = get_query_image_embedding(image_bytes)
    if not query_embedding:
        return (
            "I couldn't process that image. Please ensure it's a clear movie poster in JPG or PNG format, "
            "and try uploading again.",
            []
        )
    
    # Search for similar posters
    retrieved_movies = retrieve_movies_by_poster_similarity(query_embedding, top_k=10)
    if not retrieved_movies:
        return (
            "I couldn't find movies with visually similar posters in our database. "
            "This is a demo version with limited movie data. Try uploading a popular movie poster!",
            []
        )
    
    logger.info(f"Found {len(retrieved_movies)} similar movies")
    
    # Generate LLM response
    movie_context = format_movies_for_llm_prompt(retrieved_movies)
    system_message = "You are CineBot, a friendly movie recommendation assistant. Provide concise and engaging suggestions."
    
    prompt = f"""User has provided a movie poster. They are looking for movies with a similar visual style or implied genre/mood.

Context from movie database (these movies have visually similar posters and we have their full details):
{movie_context}

Please recommend 2-3 movie(s). Use the following format for **each** recommendation:
MOVIE: [Exact Title]
EXPLANATION: [A short (1–2 sentence) engaging explanation based on the tagline or overview.]

Example:
MOVIE: Inception
EXPLANATION: A visually stunning sci-fi thriller that dives into the world of dreams, perfect for fans of layered storytelling.

Now generate your recommendations in that format:
"""
    llm_response = get_llm_response(prompt, system_message)
    
    if not llm_response:
        # Fallback response
        top_3 = retrieved_movies[:3]
        fallback = "Here are some visually similar movies I found:\n\n"
        for movie in top_3:
            title = movie.get('title', 'Unknown')
            fallback += f"MOVIE: {title}\nEXPLANATION: This poster has a similar visual style.\n\n"
        return fallback.strip(), retrieved_movies
    
    return llm_response, retrieved_movies


def process_query(
    user_query: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    chat_history: List[Dict[str, Any]] = None
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main entry point for handling user queries.
    Orchestrates classification and delegation to the correct handler.
    Returns: (message, movies, metadata)
    """
    chat_history = chat_history or []

    if image_bytes:
        message, movies = recommend_by_poster_image(image_bytes, chat_history)
        metadata = {
            "response_type": "recommendation",
            "source": "poster_search",
            "input_mode": "image"
        }
        return message, movies, metadata

    if not user_query:
        return "Please provide a query.", [], {"response_type": "error", "source": "none", "input_mode": "none"}

    intent = classify_query_intent(user_query, chat_history)
    logger.info(f"Classified intent as: '{intent}'")

    if intent == 'graph_search':
        message, movies = handle_graph_search(user_query, chat_history)
        metadata = {
            "response_type": "answer" if not movies else "recommendation",
            "source": "graph_search",
            "input_mode": "text"
        }
        return message, movies, metadata
    elif intent == 'follow_up':
        message, movies = handle_follow_up(user_query, chat_history)
        metadata = {
            "response_type": "answer" if not movies else "recommendation",
            "source": "follow_up",
            "input_mode": "text"
        }
        return message, movies, metadata
    else:
        message, movies = handle_vector_search(user_query, chat_history)
        metadata = {
            "response_type": "recommendation",
            "source": "vector_search",
            "input_mode": "text"
        }
        return message, movies, metadata
    
if __name__ == "__main__":
    # Example usage
    example_query = "Recommend me a romantic comedy."
    example_history = [{"role": "user", "content": example_query}]
    
    response, movies = process_query(user_query=example_query, chat_history=example_history)
    print("Response:", response)
    print("Movies:", movies)
    # Note: This is just a test run; in production, this would be called from the web app or API.
    # The actual chat history and user queries would come from user interactions
