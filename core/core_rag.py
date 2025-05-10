# rag_core.py

import logging
import os
from typing import List, Dict, Optional

import torch
from PIL import Image
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel # Assuming these are still needed for query image embedding
#from langchain_community.graphs import Neo4jGraph # Or from langchain_neo4j
from langchain_neo4j import Neo4jGraph # Using the newer one

from config import settings # Your settings file
import io

NEO4J_URI = settings.NEO4J_URI
NEO4J_USERNAME = settings.NEO4J_USERNAME
NEO4J_PASSWORD = settings.NEO4J_PASSWORD
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_ENDPOINT = settings.OPENAI_ENDPOINT
TMDB_API_KEY = settings.TMDB_API_KEY
OMDB_API = settings.OMDB_API
BASE_URL = "https://api.themoviedb.org/3"
OMDB_URL = f"http://www.omdbapi.com/?apikey={OMDB_API}&"
# --- Global Initializations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
# --- OpenAI LLM Client ---
openai_llm_client = None
if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
    try:
        openai_llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAI LLM client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI LLM client: {e}")
else:
    logger.error("OpenAI API Key not found in settings.")


# --- Neo4j Connection ---
kg = None
if hasattr(settings, 'NEO4J_URI'):
    try:
        kg = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=getattr(settings, 'NEO4J_DATABASE', "neo4j")
        )
        kg.query("RETURN 1") # Test connection
        logger.info(f"Successfully connected to Neo4j at {settings.NEO4J_URI}.")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
else:
    logger.error("Neo4j URI not found in settings.")

# --- CLIP Model (for query image embedding) ---
CLIP_MODEL_NAME_CONST = "openai/clip-vit-base-patch32"
clip_model = None
clip_processor = None
DEVICE = "cpu"

try:
    logger.info("⏳ Loading CLIP processor and model...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME_CONST)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME_CONST)
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
        logger.info("✅ CUDA available. Using GPU.")
    else:
        logger.info("🧠 Using CPU for inference.")
    
    clip_model.to(DEVICE)
    logger.info(f"✅ CLIP model '{CLIP_MODEL_NAME_CONST}' successfully loaded on {DEVICE}.")

except Exception as e:
    logger.exception("❌ Failed to load CLIP model and processor.")



# --- Embedding and Retrieval Functions (Placeholders - you should have these implemented) ---

def get_text_embedding_openai(text_to_embed: str) -> Optional[List[float]]:
    if not openai_llm_client: return None
    try:
        response = openai_llm_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text_to_embed
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating OpenAI embedding for text: {e}")
        return None

def retrieve_movies_by_text_similarity(query_text: str, top_k: int = 5) -> List[Dict]:
    if not kg: return []
    query_embedding = get_text_embedding_openai(query_text)
    if not query_embedding:
        logger.warning("Could not generate query embedding. Cannot retrieve similar movies.")
        return []
    # Ensure 'movie_tagline_embeddings' index exists with 1536 dimensions
    cypher_query = """
    CALL db.index.vector.queryNodes('movie_tagline_embeddings', $top_k, $query_embedding)
    YIELD node AS similarMovie, score
    RETURN similarMovie.tmdb_id AS tmdb_id,
           similarMovie.title AS title,
           similarMovie.tagline AS tagline,
           similarMovie.overview AS overview,
           similarMovie.poster_url AS poster_url,
           score
    ORDER BY score DESC
    """
    try:
        results = kg.query(cypher_query, params={"top_k": top_k, "query_embedding": query_embedding})
        return results if results else []
    except Exception as e:
        logger.error(f"Error querying Neo4j for text similarity: {e}")
        return []

# def get_query_image_embedding(image_bytes: bytes) -> Optional[List[float]]: # Takes bytes now
#     if not clip_model or not clip_processor:
#         logger.error("CLIP model or processor not loaded.")
#         return None
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # Load from bytes
#         #inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
#         inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

#         with torch.no_grad():
#             image_features = clip_model.get_image_features(**inputs)
#             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
#             return image_features[0].cpu().tolist()
#     except Exception as e:
#         logger.error(f"Error generating embedding for query image: {e}", exc_info=True)
#         return None
# def get_query_image_embedding(image_bytes: bytes) -> Optional[List[float]]:
#     if not clip_model or not clip_processor:
#         logger.error("CLIP model or processor not loaded.")
#         return None
#     if not image_bytes:
#         logger.warning("No image bytes provided for embedding.")
#         return None
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#         # ===> RE-ADD PADDING AND TRUNCATION HERE <===
#         inputs = clip_processor(
#             images=image,
#             return_tensors="pt",
#             padding=True,       # Or padding="max_length"
#             truncation=True
#         ).to(DEVICE)

#         with torch.no_grad():
#             image_features = clip_model.get_image_features(**inputs)
#             # Normalize the image features
#             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
#         return image_features[0].cpu().tolist()
#     except ValueError as ve: # Catch ValueError specifically
#         logger.error(f"ValueError during image processing/tensor creation: {ve}", exc_info=True)
#         if "padding=True" in str(ve):
#             logger.error("This likely means the CLIPProcessor needs explicit padding/truncation for this image or batch setup.")
#         return None
#     except Exception as e:
#         logger.error(f"Unexpected error generating embedding for query image: {e}", exc_info=True)
#         return None
def get_query_image_embedding(image_bytes: bytes) -> Optional[List[float]]:
    if not clip_model or not clip_processor:
        logger.error("CLIP model or processor not loaded.")
        return None
    if not image_bytes:
        logger.warning("No image bytes provided for embedding.")
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ✅ Do NOT pass padding/truncation here — image only
        inputs = clip_processor(
            images=image,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features[0].cpu().tolist()

    except Exception as e:
        logger.error(f"Error generating embedding for query image: {e}", exc_info=True)
        return None

def retrieve_movies_by_text_similarity(query_text: str, top_k: int = 5) -> List[Dict]:
    if not kg: return []
    query_embedding = get_text_embedding_openai(query_text)
    if not query_embedding:
        logger.warning("Could not generate query embedding for text similarity.")
        return []
    cypher_query = """
    CALL db.index.vector.queryNodes('movie_tagline_embeddings', $top_k, $query_embedding)
    YIELD node AS similarMovie, score
    RETURN similarMovie.tmdb_id AS tmdb_id,
           similarMovie.title AS title,
           similarMovie.tagline AS tagline,
           similarMovie.overview AS overview,
           similarMovie.poster_url AS poster_url,   // Ensure this is fetched
           similarMovie.trailer_url AS trailer_url, // Ensure this is fetched
           score
    ORDER BY score DESC
    """
    try:
        results = kg.query(cypher_query, params={"top_k": top_k, "query_embedding": query_embedding})
        logger.debug(f"Text similarity retrieval found {len(results) if results else 0} movies.")
        return results if results else []
    except Exception as e:
        logger.error(f"Error querying Neo4j for text similarity: {e}")
        return []

# Similarly for retrieve_movies_by_poster_similarity
def retrieve_movies_by_poster_similarity(query_image_embedding: List[float], top_k: int = 5) -> List[Dict]:
    if not kg or not query_image_embedding: return []
    cypher_query = """
    CALL db.index.vector.queryNodes('movie_poster_embeddings', $top_k, $query_poster_embedding)
    YIELD node AS similarMovie, score
    RETURN similarMovie.tmdb_id AS tmdb_id,
           similarMovie.title AS title,
           similarMovie.tagline AS tagline,
           similarMovie.overview AS overview,
           similarMovie.poster_url AS poster_url,   // Ensure this is fetched
           similarMovie.trailer_url AS trailer_url, // Ensure this is fetched
           score
    ORDER BY score DESC
    """
    try:
        results = kg.query(cypher_query, params={"top_k": top_k, "query_poster_embedding": query_image_embedding})
        logger.debug(f"Poster similarity retrieval found {len(results) if results else 0} movies.")
        return results if results else []
    except Exception as e:
        logger.error(f"Error querying Neo4j for poster similarity: {e}")
        return []
# --- LLM Interaction ---
def get_llm_recommendation(prompt_text: str) -> str:
    if not openai_llm_client:
        return "LLM client not initialized. Please check API key."
    try:
        completion = openai_llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a friendly and insightful movie recommendation assistant. You provide concise (2-3 sentences per movie) and engaging suggestions."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting response from LLM: {e}", exc_info=True)
        return "I'm sorry, I encountered an error while generating a recommendation."

# --- Prompt Formatting ---
def format_movies_for_llm_prompt(movies: List[Dict], query_type: str) -> str:
    if not movies:
        return "No relevant movies were found in the database to provide context."
    context_str = f"Based on the user's {query_type} query, here are some potentially relevant movies found in our database (titles, taglines, and overview snippets):\n\n"
    for i, movie in enumerate(movies[:3]): # Limit context for LLM to top 3-5
        context_str += f"{i+1}. Title: {movie.get('title', 'N/A')}\n"
        if movie.get('tagline'):
            context_str += f"   Tagline: {movie.get('tagline')}\n"
        if movie.get('overview'):
            context_str += f"   Overview: {movie.get('overview', '')[:200]}...\n"
        context_str += "\n"
    return context_str

# --- Core RAG Logic Functions ---
# def recommend_by_text(user_text_query: str, top_k_retrieval: int = 5, num_recommendations: int = 2) -> str:
#     logger.info(f"RAG - Text Query: '{user_text_query}'")
#     retrieved_movies = retrieve_movies_by_text_similarity(user_text_query, top_k=top_k_retrieval)
#     if not retrieved_movies:
#         return "I couldn't find movies closely matching your description. Could you try rephrasing?"
#     movie_context = format_movies_for_llm_prompt(retrieved_movies, "text description")
#     prompt = f"""User is looking for movies based on the description: "{user_text_query}"

# Context from movie database:
# {movie_context}

# Based on this context, please recommend {num_recommendations} movie(s) that best fit the user's description.
# For each movie: State its title and briefly (1-2 sentences) explain why it's a good match, using its tagline or overview.
# Make your response engaging. If none of the context movies seem a strong fit, say so.
# """
#     return get_llm_recommendation(prompt)

# import io # Required for BytesIO

# def recommend_by_poster_image(query_image_bytes: bytes, top_k_retrieval: int = 5, num_recommendations: int = 2) -> str:
#     logger.info("RAG - Image Query received.")
#     query_embedding = get_query_image_embedding(query_image_bytes)
#     if not query_embedding:
#         return "I'm sorry, I couldn't process the poster image you provided."
#     retrieved_movies = retrieve_movies_by_poster_similarity(query_embedding, top_k=top_k_retrieval)
#     if not retrieved_movies:
#         return "I couldn't find movies with posters visually similar to the one you provided."
#     movie_context = format_movies_for_llm_prompt(retrieved_movies, "poster image")
#     prompt = f"""User has provided a movie poster. They are looking for movies with a similar visual style or implied genre/mood.

# Context from movie database (movies with visually similar posters):
# {movie_context}

# Based on this context, please recommend {num_recommendations} movie(s).
# For each movie: State its title and briefly (1-2 sentences) describe the movie using its tagline or overview.
# Make your response engaging.
# """
#     return get_llm_recommendation(prompt)

# rag_core.py

def recommend_by_text(user_text_query: str, top_k_retrieval: int = 5, num_recommendations: int = 2) -> tuple[str, List[Dict]]:
    logger.info(f"RAG - Text Query: '{user_text_query}'")
    # This list will contain full details including poster_url and trailer_url
    initial_retrieved_movies = retrieve_movies_by_text_similarity(user_text_query, top_k=top_k_retrieval)

    if not initial_retrieved_movies:
        return "I couldn't find movies closely matching your description. Could you try rephrasing?", []

    movie_context_for_llm = format_movies_for_llm_prompt(initial_retrieved_movies, "text description")
    
    prompt = f"""User is looking for movies based on the description: "{user_text_query}"

Context from movie database (these movies have similar taglines/overviews and we have their full details):
{movie_context_for_llm}

Based on this context, please recommend {num_recommendations} movie(s) that best fit the user's description.
For EACH recommended movie, provide:
1. The EXACT TITLE of the movie as listed in the context.
2. A short, engaging EXPLANATION (1-2 sentences) for why it's a good match, drawing from its tagline or overview.

Format your response clearly for each movie. For example:
MOVIE: [EXACT_TITLE_1_FROM_CONTEXT]
EXPLANATION: [EXPLANATION_1]
""" # Emphasize using EXACT title
    
    llm_explanation_text = get_llm_recommendation(prompt)
    return llm_explanation_text, initial_retrieved_movies

def recommend_by_poster_image(query_image_bytes: bytes, top_k_retrieval: int = 5, num_recommendations: int = 2) -> tuple[str, List[Dict]]:
    logger.info("RAG - Image Query received.")
    query_embedding = get_query_image_embedding(query_image_bytes)

    if not query_embedding:
        return "I'm sorry, I couldn't process the poster image you provided.", []

    initial_retrieved_movies = retrieve_movies_by_poster_similarity(query_embedding, top_k=top_k_retrieval)

    if not initial_retrieved_movies:
        return "I couldn't find movies with posters visually similar to the one you provided.", []

    movie_context_for_llm = format_movies_for_llm_prompt(initial_retrieved_movies, "poster image")

    prompt = f"""
User has provided a movie poster. They are looking for movies with a similar visual style or implied genre/mood.

Context from movie database (these movies have visually similar posters and we have their full details):
{movie_context_for_llm}

Please recommend {num_recommendations} movie(s). Use the following format for **each** recommendation:
MOVIE: [Exact Title]
EXPLANATION: [A short (1–2 sentence) engaging explanation based on the tagline or overview.]

Example:
MOVIE: Inception
EXPLANATION: A visually stunning sci-fi thriller that dives into the world of dreams, perfect for fans of layered storytelling.

Now generate your recommendations in that format:
"""

    llm_explanation_text = get_llm_recommendation(prompt)
    return llm_explanation_text, initial_retrieved_movies
