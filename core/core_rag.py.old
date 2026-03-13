#core_rag.py

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
from utils.poster_filter import is_valid_movie_poster

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

# --- Intent Classification --
def classify_query_intent(user_query: str, chat_history: List[Dict]) -> str:
    if not openai_client: return "vector_search"
    
    # Heuristic: If the last assistant message had no context, treat next as graph_search
    if chat_history and chat_history[-1]["role"] == "assistant" and not chat_history[-1].get("context"):
       logger.info("[IntentClassifier] Last assistant message had no context → Forcing intent: graph_search")
       return "graph_search"


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
    logger.info("Handling query as a FOLLOW-UP")

    if not chat_history:
        return "There is no previous conversation to follow up on.", []

    last_bot_message = chat_history[-1]
    previous_context_movies = last_bot_message.get("context", [])

    # 🚨 Reroute early if no real movie data
    if not previous_context_movies or not any("title" in m for m in previous_context_movies):
        logger.info("No movie context or invalid context found. Rerouting to graph_search.")
        return handle_graph_search(user_query, chat_history)

    # 🧠 Proceed with LLM-based follow-up on movie list
    movie_context = format_movies_for_llm_prompt(previous_context_movies)
    history_context = format_chat_history_for_llm(chat_history[:-1])

    system_message = (
        "You are CineBot. Answer the user's follow-up question based *only* on the previous list of movies "
        "provided in the CONTEXT. Be concise."
    )

    prompt = f"""{history_context}
I previously recommended the following movies:
CONTEXT:
{movie_context}

Now, the user has a follow-up question: "{html.escape(user_query)}"

TASK: Answer the user's question based on the information in the CONTEXT. If you cannot answer from the context, say so.
"""

    answer = get_llm_response(prompt, system_message)

    # ✅ Reroute if LLM still fails
    if (
        "cannot answer" in answer.lower()
        or "don't know" in answer.lower()
        or "no context" in answer.lower()
        or "i'm sorry" in answer.lower()
    ):
        logger.info("LLM failed to answer follow-up. Fallback to graph search.")
        return handle_graph_search(user_query, chat_history)

    return answer, previous_context_movies


def recommend_by_poster_image(image_bytes: bytes, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
    """Generates recommendations from a poster image."""
    logger.info("Handling query with IMAGE SEARCH")
        # ---- 1. Early rejection if the picture is NOT a movie poster ----
    if not is_valid_movie_poster(image_bytes):
        logger.info("Poster validator says image is NOT a movie poster.")
        return (
            "That image doesn’t look like a film poster I recognise. "
            "Please try uploading a clear movie poster.",[]
        )
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
    
if __name__ == "__main__":
    # Example usage
    example_query = "Recommend me a romantic comedy."
    example_history = [{"role": "user", "content": example_query}]
    
    response, movies = process_query(user_query=example_query, chat_history=example_history)
    print("Response:", response)
    print("Movies:", movies)
    # Note: This is just a test run; in production, this would be called from the web app or API.
    # The actual chat history and user queries would come from user interactions