# # core_rag.py

# import logging
# import os
# import io
# from typing import List, Dict, Optional, Tuple
# import html

# import torch
# from PIL import Image
# from openai import OpenAI
# from transformers import CLIPProcessor, CLIPModel
# from langchain_neo4j import Neo4jGraph

# from config import settings

# # --- Global Initializations ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # --- OpenAI LLM Client ---
# openai_llm_client = None
# if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
#     try:
#         openai_llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
#         logger.info("OpenAI LLM client initialized.")
#     except Exception as e:
#         logger.error(f"Failed to initialize OpenAI LLM client: {e}")
# else:
#     logger.error("OpenAI API Key not found in settings.")

# # --- Neo4j Connection ---
# kg = None
# if hasattr(settings, 'NEO4J_URI'):
#     try:
#         kg = Neo4jGraph(
#             url=settings.NEO4J_URI,
#             username=settings.NEO4J_USERNAME,
#             password=settings.NEO4J_PASSWORD,
#             database=getattr(settings, 'NEO4J_DATABASE', "neo4j")
#         )
#         kg.query("RETURN 1")
#         logger.info(f"Successfully connected to Neo4j at {settings.NEO4J_URI}.")
#     except Exception as e:
#         logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
# else:
#     logger.error("Neo4j URI not found in settings.")

# # --- CLIP Model ---
# CLIP_MODEL_NAME_CONST = "openai/clip-vit-base-patch32"
# clip_model = None
# clip_processor = None
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# try:
#     logger.info(f"⏳ Loading CLIP processor and model on {DEVICE}...")
#     clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME_CONST)
#     clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME_CONST).to(DEVICE)
#     logger.info(f"✅ CLIP model '{CLIP_MODEL_NAME_CONST}' successfully loaded.")
# except Exception as e:
#     logger.exception("❌ Failed to load CLIP model and processor.")

# # --- Embedding Functions ---
# def get_text_embedding_openai(text_to_embed: str) -> Optional[List[float]]:
#     if not openai_llm_client: return None
#     try:
#         response = openai_llm_client.embeddings.create(model="text-embedding-ada-002", input=text_to_embed)
#         return response.data[0].embedding
#     except Exception as e:
#         logger.error(f"Error generating OpenAI embedding for text: {e}")
#         return None

# def get_query_image_embedding(image_bytes: bytes) -> Optional[List[float]]:
#     if not clip_model or not clip_processor or not image_bytes:
#         logger.error("CLIP model/processor not loaded or no image bytes provided.")
#         return None
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
#         with torch.no_grad():
#             image_features = clip_model.get_image_features(**inputs)
#             image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
#         return image_features[0].cpu().tolist()
#     except Exception as e:
#         logger.error(f"Error generating embedding for query image: {e}", exc_info=True)
#         return None

# # --- Retrieval Functions ---
# def retrieve_movies_by_text_similarity(query_text: str, top_k: int = 5) -> List[Dict]:
#     if not kg: return []
#     query_embedding = get_text_embedding_openai(query_text)
#     if not query_embedding: return []
#     cypher_query = """
#     CALL db.index.vector.queryNodes('movie_tagline_embeddings', $top_k, $query_embedding)
#     YIELD node AS m, score
#     RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.tagline AS tagline,
#            m.overview AS overview, m.poster_url AS poster_url, m.trailer_url AS trailer_url, score
#     ORDER BY score DESC
#     """
#     try:
#         return kg.query(cypher_query, params={"top_k": top_k, "query_embedding": query_embedding}) or []
#     except Exception as e:
#         logger.error(f"Error querying Neo4j for text similarity: {e}")
#         return []

# def retrieve_movies_by_poster_similarity(query_image_embedding: List[float], top_k: int = 5) -> List[Dict]:
#     if not kg or not query_image_embedding: return []
#     cypher_query = """
#     CALL db.index.vector.queryNodes('movie_poster_embeddings', $top_k, $query_embedding)
#     YIELD node AS m, score
#     RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.tagline AS tagline,
#            m.overview AS overview, m.poster_url AS poster_url, m.trailer_url AS trailer_url, score
#     ORDER BY score DESC
#     """
#     try:
#         return kg.query(cypher_query, params={"top_k": top_k, "query_embedding": query_image_embedding}) or []
#     except Exception as e:
#         logger.error(f"Error querying Neo4j for poster similarity: {e}")
#         return []

# # --- LLM Interaction & Prompt Formatting ---
# def get_llm_recommendation(prompt_text: str) -> str:
#     if not openai_llm_client: return "LLM client not initialized."
#     try:
#         completion = openai_llm_client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are CineBot, a friendly and insightful movie recommender. You give concise (1-2 sentence) explanations for your choices. You MUST format recommendations using 'MOVIE: Title' and 'EXPLANATION: text' pairs."},
#                 {"role": "user", "content": prompt_text}
#             ],
#             temperature=0.7
#         )
#         return completion.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error(f"Error getting LLM response: {e}", exc_info=True)
#         return "I'm sorry, I encountered an error while thinking of a recommendation."

# def format_movies_for_llm_prompt(movies: List[Dict]) -> str:
#     if not movies: return "No relevant movies were found in the database."
#     context_parts = []
#     for movie in movies[:5]:
#         title = html.escape(movie.get('title', 'N/A'))
#         tagline = html.escape(movie.get('tagline', ''))
#         overview = html.escape(movie.get('overview', ''))[:250]
#         movie_str = f"--- Movie: {title} ---\nTagline: {tagline}\nOverview: {overview}\n---"
#         context_parts.append(movie_str)
#     return "\n\n".join(context_parts)

# def format_chat_history(chat_history: List[Dict]) -> str:
#     if not chat_history: return "This is the first message in the conversation."
#     history_str = "Here is the conversation history so far:\n"
#     for message in chat_history:
#         role = "User" if message["role"] == "user" else "CineBot"
#         history_str += f"{role}: {message['content']}\n"
#     return history_str

# # --- Conversational Logic ---
# def route_user_query(user_query: str, chat_history: List[Dict]) -> str:
#     if not chat_history: return "recommendation"
#     history_context = format_chat_history(chat_history)
#     prompt = f"""Classify the user's intent based on their latest message and the conversation history.
# Categories: "recommendation" (asking for new movies) or "question" (asking about a previous recommendation).
# {history_context}
# LATEST USER QUERY: "{html.escape(user_query)}"
# Intent:"""
#     try:
#         completion = openai_llm_client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10)
#         intent = completion.choices[0].message.content.strip().lower()
#         logger.info(f"User intent classified as: '{intent}'")
#         return "question" if "question" in intent else "recommendation"
#     except Exception as e:
#         logger.error(f"Error classifying user intent: {e}")
#         return "recommendation"

# def answer_follow_up_question(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
#     history_context = format_chat_history(chat_history)
#     prompt = f"""You are a movie assistant. Answer the user's follow-up question based ONLY on the provided conversation history.
# {history_context}
# The user's new question is: "{html.escape(user_query)}"
# Please answer concisely. If the answer is not in the history, state that you don't have the information from the previous chat."""
#     answer = get_llm_recommendation(prompt)
#     return answer, []

# # --- Core RAG Logic Functions ---
# def recommend_by_text(user_query: str, chat_history: List[Dict], top_k_retrieval: int = 7, num_recommendations: int = 3) -> Tuple[str, List[Dict]]:
#     intent = route_user_query(user_query, chat_history)
#     if intent == "question":
#         return answer_follow_up_question(user_query, chat_history)
    
#     initial_retrieved_movies = retrieve_movies_by_text_similarity(user_query, top_k=top_k_retrieval)
#     if not initial_retrieved_movies:
#         return "I couldn't find movies closely matching your description. Could you try rephrasing?", []

#     movie_context_for_llm = format_movies_for_llm_prompt(initial_retrieved_movies)
#     history_context = format_chat_history(chat_history)
    
#     prompt = f"""{history_context}
# USER'S LATEST REQUEST: "{html.escape(user_query)}"
# CONTEXT (Movies found in the database based on the latest request):
# {movie_context_for_llm}
# TASK: Based on the request and history, select the {num_recommendations} best movies from the CONTEXT.
# For EACH selected movie, respond in the following format exactly:
# MOVIE: [The EXACT movie title from the context]
# EXPLANATION: [A 1-2 sentence explanation of why this is a good match.]"""
    
#     llm_explanation_text = get_llm_recommendation(prompt)
#     return llm_explanation_text, initial_retrieved_movies

# def recommend_by_poster_image(query_image_bytes: bytes, top_k_retrieval: int = 5, num_recommendations: int = 3) -> tuple[str, List[Dict]]:
#     logger.info("RAG - Image Query received.")
#     query_embedding = get_query_image_embedding(query_image_bytes)
#     if not query_embedding:
#         return "I'm sorry, I couldn't process the poster image you provided.", []

#     initial_retrieved_movies = retrieve_movies_by_poster_similarity(query_embedding, top_k=top_k_retrieval)
#     if not initial_retrieved_movies:
#         return "I couldn't find movies with posters visually similar to the one you provided.", []

#     # FIXED: This call now correctly passes only one argument.
#     movie_context_for_llm = format_movies_for_llm_prompt(initial_retrieved_movies)

#     prompt = f"""User has provided a movie poster and wants recommendations with a similar visual style or mood.
# CONTEXT (Movies with visually similar posters):
# {movie_context_for_llm}
# TASK: Recommend {num_recommendations} movie(s) from the CONTEXT. For EACH, provide:
# MOVIE: [Exact Title from Context]
# EXPLANATION: [A short, engaging explanation based on its tagline or overview.]"""

#     llm_explanation_text = get_llm_recommendation(prompt)
#     return llm_explanation_text, initial_retrieved_movies

# core_rag_refactored.py

import logging
import os
import io
from typing import List, Dict, Optional, Tuple, Any
import html

import torch
from PIL import Image
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel

# --- NEW: Import LangChain components for Text-to-Cypher ---
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
# ---

from langchain_neo4j import Neo4jGraph
from config import settings
from langchain.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph

# --- Global Initializations (largely the same) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            database=getattr(settings, 'NEO4J_DATABASE', "neo4j")
        )
        kg.refresh_schema() # Important for LangChain
        logger.info(f"Successfully connected to Neo4j and refreshed schema.")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
else:
    logger.error("Neo4j URI not found in settings.")
print(kg.schema)


# --- CLIP Model (same) ---
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
    if not clip_model or not clip_processor or not image_bytes: return None
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
def get_llm_response(prompt_text: str, system_message: str) -> str:
    if not openai_client: return "LLM client not initialized."
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini", # Using a newer, cheaper model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.5
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}", exc_info=True)
        return "I'm sorry, I encountered an error. Please try again."

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

# ------------------------------------------------------------------------------------
# --- NEW: Core RAG Logic - The Router and Handlers ---
# ------------------------------------------------------------------------------------

# def classify_query_intent(user_query: str, chat_history: List[Dict]) -> str:
#     """
#     Classifies the user's intent to decide which tool to use.
#     """
#     if not openai_client: return "vector_search" # Default fallback
    
#     # If the history is empty, it must be a new search, not a follow-up
#     if not chat_history:
#         history_context = "This is the first message."
#     else:
#         history_context = f"The last message from the bot was: '{chat_history[-1]['content']}'"

#     prompt = f"""
# You are an expert query classifier for a movie recommendation chatbot.
# Your task is to classify the user's latest query into one of three categories based on the query and conversation history:

# 1.  `graph_search`: The user is asking a specific, factual question that can be answered with a database query.
#     Examples: "Who directed Inception?", "List movies with Tom Hanks", "When was The Matrix released?"

# 2.  `vector_search`: The user is asking for a recommendation based on a vague description, mood, or theme.
#     Examples: "I want to watch a funny space movie", "Suggest something romantic and exciting", "movies like Blade Runner"

# 3.  `follow_up`: The user is asking a question about the movies that were just recommended in the previous turn.
#     Examples: "Tell me more about the second one", "Which one is a comedy?", "Do you have anything else by that director?"

# Conversation context:
# {history_context}

# User's latest query: "{html.escape(user_query)}"

# Classification:
# """
#     try:
#         completion = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_tokens=20
#         )
#         intent = completion.choices[0].message.content.strip().lower()
#         # Clean up the response to get one of the three keys
#         if "graph_search" in intent: return "graph_search"
#         if "vector_search" in intent: return "vector_search"
#         if "follow_up" in intent: return "follow_up"

#         logger.warning(f"Could not reliably classify intent '{intent}', defaulting to vector_search.")
#         return "vector_search" # Fallback
#     except Exception as e:
#         logger.error(f"Error classifying user intent: {e}")
#         return "vector_search" # Fallback on error


# def handle_vector_search(user_query: str, chat_history: List[Dict], top_k: int = 7, num_rec: int = 3) -> Tuple[str, List[Dict]]:
#     """Handles semantic search using vector similarity."""
#     logger.info("Handling query with VECTOR SEARCH")
#     retrieved_movies = retrieve_movies_by_text_similarity(user_query, top_k=top_k)
#     if not retrieved_movies:
#         return "I couldn't find any movies matching that description. Could you try being more specific or rephrasing?", []

#     movie_context = format_movies_for_llm_prompt(retrieved_movies)
#     history_context = format_chat_history_for_llm(chat_history)

#     system_message = "You are CineBot, a movie recommender. Select the best movies from the context provided and explain why in 1-2 sentences. Format your response exactly as: MOVIE: [Title]\nEXPLANATION: [Your text]"
#     prompt = f"""{history_context}
# Based on the user's request for "{html.escape(user_query)}", I have found the following movies:
# CONTEXT:
# {movie_context}

# TASK: Select the {num_rec} best movies from the CONTEXT. For EACH, respond in the required format.
# """
#     llm_response = get_llm_response(prompt, system_message)
#     return llm_response, retrieved_movies


# # def handle_graph_search(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
# #     """Handles factual questions using Text-to-Cypher."""
# #     logger.info("Handling query with GRAPH SEARCH (Text-to-Cypher)")
# #     if not kg or not settings.OPENAI_API_KEY:
# #         return "My graph search functionality is not configured. Please ask for a recommendation instead.", []

# #     # This chain generates and executes a Cypher query
# #     cypher_chain = GraphCypherQAChain.from_llm(
# #        graph=kg,
# #     cypher_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
# #     qa_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
# #     validate_cypher=True,
# #     verbose=True #set false for production
# #     )
    
# #     try:
# #         # The chain returns a dictionary, we are interested in the 'result'
# #         result = cypher_chain.invoke({"query": user_query})
# #         answer = result.get("result", "I couldn't find a direct answer in the database.")
        
# #         # We return an empty list for context, as this is a direct answer, not a recommendation list
# #         return answer, []
# #     except Exception as e:
# #         logger.error(f"Error during GraphCypherQAChain execution: {e}")
# #         return "I had trouble querying the database for that information. Please try rephrasing your question.", []

# # In your file: core/core_rag.py

# # (Make sure these imports are at the top of the file)
# from langchain.prompts.prompt import PromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain.chains import GraphCypherQAChain
# # ... other imports ...


# def handle_graph_search(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
#     """Handles factual questions using Text-to-Cypher with a custom, few-shot prompt."""
#     logger.info("Handling query with GRAPH SEARCH (Text-to-Cypher)")
#     if not kg or not settings.OPENAI_API_KEY:
#         return "My graph search functionality is not configured.", []

#     # --- NEW, IMPROVED FEW-SHOT PROMPT ---
#     # This template includes examples to guide the LLM.
#     # IMPORTANT: You may need to adjust the examples to match your EXACT graph schema.
#     CYPHER_GENERATION_TEMPLATE = """
#     Task: Generate Cypher statement to query a graph database.
#     Instructions:
#     Use only the provided relationship types and properties in the schema.
#     Do not use any other relationship types or properties that are not provided.
#     Do not return any explanations or apologies.
#     Your response must be in a markdown code block with the "cypher" tag.

#     Schema:
#     {schema}

#     ---
#     Here are some examples of questions and their corresponding Cypher queries.

#     Question: Who directed the movie The Matrix?
#     Cypher:
#     ```cypher
#     MATCH (p:Person)-[:DIRECTED]->(m:Movie {{title: 'The Matrix'}})
#     RETURN p.name
#     ```

#     Question: What movies did Tom Hanks act in?
#     Cypher:
#     ```cypher
#     MATCH (p:Person {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie)
#     RETURN m.title
#     ```

#     Question: List actors from the movie Forrest Gump.
#     Cypher:
#     ```cypher
#     MATCH (p:Person)-[:ACTED_IN]->(m:Movie {{title: 'Forrest Gump'}})
#     RETURN p.name
#     ```
#     ---

#     Now, generate the Cypher statement for this question:
#     Question: {question}
#     """

#     cypher_prompt = PromptTemplate(
#         input_variables=["schema", "question"],
#         template=CYPHER_GENERATION_TEMPLATE
#     )
#     # --- END OF NEW PROMPT DEFINITION ---

#     try:
#         # We use the from_llm constructor which is designed for this purpose.
#         cypher_chain = GraphCypherQAChain.from_llm(
#             cypher_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
#             qa_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
#             graph=kg,
#             verbose=True,
#             cypher_prompt=cypher_prompt # Pass the new, powerful prompt
#         )

#         result = cypher_chain.invoke({"query": user_query})
#         answer = result.get("result", "I couldn't find a direct answer in the database.")
#         return answer, []

#     except Exception as e:
#         logger.error(f"Error during GraphCypherQAChain execution: {e}", exc_info=True)
#         return "I had trouble querying the database for that information. Please try rephrasing your question.", []
# def handle_follow_up(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
#     """Handles follow-up questions based on the last interaction's context."""
#     logger.info("Handling query as a FOLLOW-UP")
#     if not chat_history:
#         return "There is no previous conversation to follow up on. Please ask for a new recommendation.", []

#     last_bot_message = chat_history[-1]
#     # IMPORTANT: We retrieve the context from our stateful chat history
#     previous_context_movies = last_bot_message.get("context", [])

#     if not previous_context_movies:
#         return "I don't have a specific list of movies from our last chat to discuss. Let's start a new search!", []
    
#     movie_context = format_movies_for_llm_prompt(previous_context_movies)
#     history_context = format_chat_history_for_llm(chat_history[:-1]) # History without the last bot message

#     system_message = "You are CineBot. Answer the user's follow-up question based *only* on the previous list of movies provided in the CONTEXT. Be concise."
#     prompt = f"""{history_context}
# I previously recommended the following movies:
# CONTEXT:
# {movie_context}

# Now, the user has a follow-up question: "{html.escape(user_query)}"

# TASK: Answer the user's question based on the information in the CONTEXT. If you cannot answer from the context, say so.
# """
#     answer = get_llm_response(prompt, system_message)
#     # We return the same context so the user can ask another follow-up
#     return answer, previous_context_movies


# def recommend_by_poster_image(image_bytes: bytes, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
#     """Generates recommendations from a poster image."""
#     logger.info("Handling query with IMAGE SEARCH")
#     query_embedding = get_query_image_embedding(image_bytes)
#     if not query_embedding:
#         return "I couldn't process that image. Please try another one.", []
    
#     retrieved_movies = retrieve_movies_by_poster_similarity(query_embedding, top_k=5)
#     if not retrieved_movies:
#         return "I couldn't find movies with visually similar posters.", []
    
#     movie_context = format_movies_for_llm_prompt(retrieved_movies)
    
#     system_message = "You are CineBot. The user uploaded a poster. Recommend movies from the context with similar visual styles. Format your response exactly as: MOVIE: [Title]\nEXPLANATION: [Your text]"
#     prompt = f"""A user uploaded a movie poster. I found these movies with similar-looking posters:
# CONTEXT:
# {movie_context}

# TASK: Recommend 3 movies from the CONTEXT. Explain why based on their overview or tagline.
# """
#     llm_response = get_llm_response(prompt, system_message)
#     # We return the retrieved movies as context for potential follow-ups
#     return llm_response, retrieved_movies


# # ------------------------------------------------------------------------------------
# # --- NEW: Main Orchestrator Function ---
# # ------------------------------------------------------------------------------------

# def process_query(
#     user_query: Optional[str] = None,
#     image_bytes: Optional[bytes] = None,
#     chat_history: List[Dict[str, Any]] = None
# ) -> Tuple[str, List[Dict[str, Any]]]:
#     """
#     Main entry point for handling user queries.
#     Orchestrates classification and delegation to the correct handler.
    
#     Args:
#         user_query: The text query from the user.
#         image_bytes: The image file bytes, if uploaded.
#         chat_history: The conversation history. Each item can have 'role', 'content', and 'context'.

#     Returns:
#         A tuple containing:
#         - The natural language response for the user.
#         - The list of movies that were used as context for this turn.
#     """
#     chat_history = chat_history or []

#     # 1. Handle image query first, as it's a distinct modality
#     if image_bytes:
#         # Image queries are always treated as new vector searches, not follow-ups
#         return recommend_by_poster_image(image_bytes, chat_history)

#     if not user_query:
#         return "Please provide a query.", []

#     # 2. Classify the intent of the text query
#     intent = classify_query_intent(user_query, chat_history)
#     logger.info(f"Classified intent as: '{intent}'")

#     # 3. Delegate to the appropriate handler based on intent
#     if intent == 'graph_search':
#         return handle_graph_search(user_query, chat_history)
#     elif intent == 'follow_up':
#         return handle_follow_up(user_query, chat_history)
#     else: # Default to 'vector_search'
#         return handle_vector_search(user_query, chat_history)

# In your file: core/core_rag.py
# Paste this entire block to replace your existing functions

def classify_query_intent(user_query: str, chat_history: List[Dict]) -> str:
    if not openai_client: return "vector_search"

    history_context = (
        "This is the first message." if not chat_history
        else f"The last message from the bot was: '{chat_history[-1]['content']}'"
    )

    prompt = f"""
You are an intent classifier for a movie chatbot.

Your job is to classify the user's latest message into ONE of exactly these categories:
- `graph_search`: For factual questions using named entities (e.g., 'Who directed Interstellar?', 'List movies with Tom Cruise')
- `vector_search`: For recommendations based on mood, theme, or vague descriptions (e.g., 'Give me something romantic and funny')
- `follow_up`: For follow-up questions about previous results (e.g., 'Tell me more about the second one', 'Which one was a comedy?')

Rules:
- Always return just one of: graph_search, vector_search, follow_up
- Be strict. If it's a WH-question (who, what, when, where, how) and involves a movie/person title — it's graph_search.

Conversation context:
{history_context}

User's latest query: "{html.escape(user_query)}"

Just return one of: graph_search, vector_search, follow_up.
"""

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
        logger.info(f"[IntentClassifier] Query: {user_query} → Classified as: {intent}")

        if intent in {"graph_search", "vector_search", "follow_up"}:
            return intent

        logger.warning(f"Unclear intent '{intent}', defaulting to vector_search.")
        return "vector_search"
    except Exception as e:
        logger.error(f"Error classifying user intent: {e}")
        return "vector_search"


def handle_vector_search(user_query: str, chat_history: List[Dict], top_k: int = 7, num_rec: int = 3) -> Tuple[str, List[Dict]]:
    """Handles semantic search using vector similarity."""
    logger.info("Handling query with VECTOR SEARCH")
    retrieved_movies = retrieve_movies_by_text_similarity(user_query, top_k=top_k)
    if not retrieved_movies:
        return "I couldn't find any movies matching that description. Could you try rephrasing?", []

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

    Schema:
    {schema}

    ---
    Here are some examples of questions and their corresponding Cypher queries.

    Question: Who directed the movie The Matrix?
    Cypher:
    ```cypher
    MATCH (p:Person)-[:DIRECTED]->(m:Movie {{title: 'The Matrix'}})
    RETURN p.name
    ```

    Question: What movies did Tom Hanks act in?
    Cypher:
    ```cypher
    MATCH (p:Person {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie)
    RETURN m.title
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
        cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
            qa_llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
            graph=kg,
            verbose=True,
            cypher_prompt=cypher_prompt,
            allow_dangerous_requests=True
        
        )
        result = cypher_chain.invoke({"query": user_query})
        answer = result.get("result", "I couldn't find a direct answer in the database.")
        return answer, []
    except Exception as e:
        logger.error(f"Error during GraphCypherQAChain execution: {e}", exc_info=True)
        return "I had trouble querying the database for that information. Please try rephrasing.", []


def handle_follow_up(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
    """Handles follow-up questions based on the last interaction's context."""
    logger.info("Handling query as a FOLLOW-UP")
    if not chat_history:
        return "There is no previous conversation to follow up on.", []

    last_bot_message = chat_history[-1]
    previous_context_movies = last_bot_message.get("context", [])

    if not previous_context_movies:
        return "I don't have a specific list of movies from our last chat to discuss.", []
    
    movie_context = format_movies_for_llm_prompt(previous_context_movies)
    history_context = format_chat_history_for_llm(chat_history[:-1])

    system_message = "You are CineBot. Answer the user's follow-up question based *only* on the previous list of movies provided in the CONTEXT. Be concise."
    prompt = f"""{history_context}
I previously recommended the following movies:
CONTEXT:
{movie_context}

Now, the user has a follow-up question: "{html.escape(user_query)}"

TASK: Answer the user's question based on the information in the CONTEXT. If you cannot answer from the context, say so.
"""
    answer = get_llm_response(prompt, system_message)
    return answer, previous_context_movies


def recommend_by_poster_image(image_bytes: bytes, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
    """Generates recommendations from a poster image."""
    logger.info("Handling query with IMAGE SEARCH")
    query_embedding = get_query_image_embedding(image_bytes)
    if not query_embedding:
        return "I couldn't process that image. Please try another one.", []
    
    retrieved_movies = retrieve_movies_by_poster_similarity(query_embedding, top_k=5)
    if not retrieved_movies:
        return "I couldn't find movies with visually similar posters.", []
    
    movie_context = format_movies_for_llm_prompt(retrieved_movies)
    system_message = "You are CineBot. The user uploaded a poster. Recommend movies from the context with similar visual styles. Format your response exactly as: MOVIE: [Title]\nEXPLANATION: [Your text]"
    prompt = f"""A user uploaded a movie poster. I found these movies with similar-looking posters:
CONTEXT:
{movie_context}

TASK: Recommend 3 movies from the CONTEXT. Explain why based on their overview or tagline.
"""
    llm_response = get_llm_response(prompt, system_message)
    return llm_response, retrieved_movies


def process_query(
    user_query: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    chat_history: List[Dict[str, Any]] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Main entry point for handling user queries.
    Orchestrates classification and delegation to the correct handler.
    """
    chat_history = chat_history or []

    if image_bytes:
        return recommend_by_poster_image(image_bytes, chat_history)

    if not user_query:
        return "Please provide a query.", []

    intent = classify_query_intent(user_query, chat_history)
    logger.info(f"Classified intent as: '{intent}'")

    if intent == 'graph_search':
        return handle_graph_search(user_query, chat_history)
    elif intent == 'follow_up':
        return handle_follow_up(user_query, chat_history)
    else:
        return handle_vector_search(user_query, chat_history)
    
# def answer_follow_up_question(user_query: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
#     """Answers a follow-up question based on the last interaction's context."""
#     logger.info("Handling query as a FOLLOW-UP")
#     if not chat_history:
#         return "There is no previous conversation to follow up on.", []   
#     last_bot_message = chat_history[-1]
#     previous_context_movies = last_bot_message.get("context", []) 
#     if not previous_context_movies:
#         return "I don't have a specific list of movies from our last chat to discuss.", []
#     movie_context = format_movies_for_llm_prompt(previous_context_movies)
#     history_context = format_chat_history_for_llm(chat_history[:-1])  # History
#     system_message = "You are CineBot. Answer the user's follow-up question based *only* on the previous list of movies provided in the CONTEXT. Be concise."
#     prompt = f"""{history_context}
# I previously recommended the following movies:
# CONTEXT:
# {movie_context}   