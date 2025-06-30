# core_rag.py

import logging
import os
import io
from typing import List, Dict, Optional, Tuple
import html

import torch
from PIL import Image
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from langchain_neo4j import Neo4jGraph

from config import settings

# --- Global Initializations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        kg.query("RETURN 1")
        logger.info(f"Successfully connected to Neo4j at {settings.NEO4J_URI}.")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
else:
    logger.error("Neo4j URI not found in settings.")

# --- CLIP Model ---
CLIP_MODEL_NAME_CONST = "openai/clip-vit-base-patch32"
clip_model = None
clip_processor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    logger.info(f"⏳ Loading CLIP processor and model on {DEVICE}...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME_CONST)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME_CONST).to(DEVICE)
    logger.info(f"✅ CLIP model '{CLIP_MODEL_NAME_CONST}' successfully loaded.")
except Exception as e:
    logger.exception("❌ Failed to load CLIP model and processor.")

# --- Embedding Functions ---
def get_text_embedding_openai(text_to_embed: str) -> Optional[List[float]]:
    if not openai_llm_client: return None
    try:
        response = openai_llm_client.embeddings.create(model="text-embedding-ada-002", input=text_to_embed)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating OpenAI embedding for text: {e}")
        return None

def get_query_image_embedding(image_bytes: bytes) -> Optional[List[float]]:
    if not clip_model or not clip_processor or not image_bytes:
        logger.error("CLIP model/processor not loaded or no image bytes provided.")
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features[0].cpu().tolist()
    except Exception as e:
        logger.error(f"Error generating embedding for query image: {e}", exc_info=True)
        return None

# --- Retrieval Functions ---
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

# --- LLM Interaction & Prompt Formatting ---
def get_llm_recommendation(prompt_text: str) -> str:
    if not openai_llm_client: return "LLM client not initialized."
    try:
        completion = openai_llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are CineBot, a friendly and insightful movie recommender. You give concise (1-2 sentence) explanations for your choices. You MUST format recommendations using 'MOVIE: Title' and 'EXPLANATION: text' pairs."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}", exc_info=True)
        return "I'm sorry, I encountered an error while thinking of a recommendation."

def format_movies_for_llm_prompt(movies: List[Dict]) -> str:
    if not movies: return "No relevant movies were found in the database."
    context_parts = []
    for movie in movies[:5]:
        title = html.escape(movie.get('title', 'N/A'))
        tagline = html.escape(movie.get('tagline', ''))
        overview = html.escape(movie.get('overview', ''))[:250]
        movie_str = f"--- Movie: {title} ---\nTagline: {tagline}\nOverview: {overview}\n---"
        context_parts.append(movie_str)
    return "\n\n".join(context_parts)

def format_chat_history(chat_history: List[Dict]) -> str:
    if not chat_history: return "This is the first message in the conversation."
    history_str = "Here is the conversation history so far:\n"
    for message in chat_history:
        role = "User" if message["role"] == "user" else "CineBot"
        history_str += f"{role}: {message['content']}\n"
    return history_str

# --- Conversational Logic ---
def route_user_query(user_query: str, chat_history: List[Dict]) -> str:
    if not chat_history: return "recommendation"
    history_context = format_chat_history(chat_history)
    prompt = f"""Classify the user's intent based on their latest message and the conversation history.
Categories: "recommendation" (asking for new movies) or "question" (asking about a previous recommendation).
{history_context}
LATEST USER QUERY: "{html.escape(user_query)}"
Intent:"""
    try:
        completion = openai_llm_client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10)
        intent = completion.choices[0].message.content.strip().lower()
        logger.info(f"User intent classified as: '{intent}'")
        return "question" if "question" in intent else "recommendation"
    except Exception as e:
        logger.error(f"Error classifying user intent: {e}")
        return "recommendation"

def answer_follow_up_question(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
    history_context = format_chat_history(chat_history)
    prompt = f"""You are a movie assistant. Answer the user's follow-up question based ONLY on the provided conversation history.
{history_context}
The user's new question is: "{html.escape(user_query)}"
Please answer concisely. If the answer is not in the history, state that you don't have the information from the previous chat."""
    answer = get_llm_recommendation(prompt)
    return answer, []

# --- Core RAG Logic Functions ---
def recommend_by_text(user_query: str, chat_history: List[Dict], top_k_retrieval: int = 7, num_recommendations: int = 3) -> Tuple[str, List[Dict]]:
    intent = route_user_query(user_query, chat_history)
    if intent == "question":
        return answer_follow_up_question(user_query, chat_history)
    
    initial_retrieved_movies = retrieve_movies_by_text_similarity(user_query, top_k=top_k_retrieval)
    if not initial_retrieved_movies:
        return "I couldn't find movies closely matching your description. Could you try rephrasing?", []

    movie_context_for_llm = format_movies_for_llm_prompt(initial_retrieved_movies)
    history_context = format_chat_history(chat_history)
    
    prompt = f"""{history_context}
USER'S LATEST REQUEST: "{html.escape(user_query)}"
CONTEXT (Movies found in the database based on the latest request):
{movie_context_for_llm}
TASK: Based on the request and history, select the {num_recommendations} best movies from the CONTEXT.
For EACH selected movie, respond in the following format exactly:
MOVIE: [The EXACT movie title from the context]
EXPLANATION: [A 1-2 sentence explanation of why this is a good match.]"""
    
    llm_explanation_text = get_llm_recommendation(prompt)
    return llm_explanation_text, initial_retrieved_movies

def recommend_by_poster_image(query_image_bytes: bytes, top_k_retrieval: int = 5, num_recommendations: int = 3) -> tuple[str, List[Dict]]:
    logger.info("RAG - Image Query received.")
    query_embedding = get_query_image_embedding(query_image_bytes)
    if not query_embedding:
        return "I'm sorry, I couldn't process the poster image you provided.", []

    initial_retrieved_movies = retrieve_movies_by_poster_similarity(query_embedding, top_k=top_k_retrieval)
    if not initial_retrieved_movies:
        return "I couldn't find movies with posters visually similar to the one you provided.", []

    # FIXED: This call now correctly passes only one argument.
    movie_context_for_llm = format_movies_for_llm_prompt(initial_retrieved_movies)

    prompt = f"""User has provided a movie poster and wants recommendations with a similar visual style or mood.
CONTEXT (Movies with visually similar posters):
{movie_context_for_llm}
TASK: Recommend {num_recommendations} movie(s) from the CONTEXT. For EACH, provide:
MOVIE: [Exact Title from Context]
EXPLANATION: [A short, engaging explanation based on its tagline or overview.]"""

    llm_explanation_text = get_llm_recommendation(prompt)
    return llm_explanation_text, initial_retrieved_movies