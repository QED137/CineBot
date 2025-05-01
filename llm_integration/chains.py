# llm_integration/chains.py

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage # For multimodal input
from llm_integration.llm_config import get_llm # Imports the multimodal LLM getter
# Remove import of old rag_prompt if not used
# from llm_integration.prompts import rag_prompt
from llm_integration.prompts import MULTIMODAL_PROMPT # Import the simplified prompt template
import logging
import base64
import io
from PIL import Image
from typing import List, Dict, Any, Optional # Ensure typing is imported

log = logging.getLogger(__name__)

# --- Helper Function to Format Graph Context ---
def format_graph_context_for_llm(context_dict: Optional[Dict[str, Any]]) -> str:
    """Formats the retrieved graph context dictionary into a string for the LLM."""
    if not context_dict or not context_dict.get('title'):
        return "No specific movie context retrieved from the graph."

    # Truncate long lists/strings for brevity in the prompt
    actors = context_dict.get('actors', ['N/A'])
    actors_str = ', '.join(actors[:5]) + ('...' if len(actors) > 5 else '')

    directors = context_dict.get('directors', ['N/A'])
    directors_str = ', '.join(directors) # Usually fewer directors

    genres = context_dict.get('genres', ['N/A'])
    genres_str = ', '.join(genres)

    plot = context_dict.get('plot', 'N/A')
    plot_str = plot[:250] + ('...' if len(plot) > 250 else '') # Truncate plot

    parts = [
        f"- Movie Title: {context_dict.get('title', 'N/A')}",
        f"- Release Year: {context_dict.get('released', 'N/A')}",
        f"- Tagline: {context_dict.get('tagline', 'N/A')}",
        f"- Director(s): {directors_str}",
        f"- Actor(s): {actors_str}",
        f"- Genre(s): {genres_str}",
        f"- Plot Summary: {plot_str}"
    ]
    return "\n".join(parts)

# --- Helper Function to Encode Image ---
def encode_image(image_bytes: bytes) -> str:
    """Encodes image bytes to base64."""
    try:
        # Optional: Basic validation by trying to open with Pillow
        img = Image.open(io.BytesIO(image_bytes))
        log.debug(f"Encoding image of format {img.format}, size {img.size} for LLM.")
    except Exception as e:
        log.error(f"Invalid image bytes provided for encoding: {e}")
        raise ValueError("Invalid image data") from e
    return base64.b64encode(image_bytes).decode('utf-8')

# --- Function to Construct Messages for Multimodal LLM ---
def build_multimodal_rag_messages(question: str, graph_context: Optional[dict], image_bytes: Optional[bytes]) -> List:
    """
    Constructs the list of messages for the multimodal LLM, including image data
    formatted according to OpenAI's vision input requirements.
    """
    formatted_context_str = format_graph_context_for_llm(graph_context)

    # Define the structure for the message content
    content = []

    # 1. Add the primary text instruction/query
    # Combine the user's question with the formatted graph context
    text_prompt = MULTIMODAL_PROMPT.format(
        question=question,
        formatted_graph_context=formatted_context_str
    )
    content.append({"type": "text", "text": text_prompt})

    # 2. Add the image if provided
    if image_bytes:
        try:
            base64_image = encode_image(image_bytes)
            log.info("Adding base64 encoded image to LLM message content.")
            content.append({
                "type": "image_url",
                "image_url": {
                    # Use base64 data URI format required by OpenAI Vision
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    # Optional: Detail level (low, high, auto) - 'auto' is default
                    # "detail": "auto"
                 }
            })
        except ValueError as e:
             log.error(f"Skipping image in LLM input due to encoding/validation error: {e}")
        except Exception as e:
             log.error(f"Unexpected error encoding image for LLM: {e}")

    # Construct the final message list (typically a single HumanMessage for vision models)
    # Include a basic system prompt for context setting.
    system_prompt = "You are cineBoat, an expert AI assistant for movies providing information and recommendations based on text, images, and graph data."
    messages = [
        # SystemMessage(content=system_prompt), # Uncomment if needed/supported well by LangChain integration
        HumanMessage(content=content)
        ]
    # Avoid logging the full base64 string in production if possible due to length
    log.debug(f"Constructed messages for LLM (image content omitted for brevity if present)")
    return messages

# --- Function to Invoke the LLM ---
def invoke_multimodal_llm(messages: List) -> str:
    """Invokes the configured multimodal LLM with the prepared messages."""
    try:
        # Get the initialized multimodal LLM instance
        llm = get_llm()
        log.info(f"Invoking multimodal LLM ({llm.model_name})...") # Log model name
        response = llm.invoke(messages)

        # Parse the response (assuming it's a standard AI message)
        parser = StrOutputParser()
        result = parser.invoke(response)
        log.info("Received response from LLM.")
        return result
    except Exception as e:
        log.error(f"Error invoking multimodal LLM: {e}", exc_info=True)
        # Provide a user-friendly error message
        return "Sorry, I encountered an error while communicating with the AI model. Please check the logs or try again later."


# --- Keep the old text-only RAG chain function if needed for comparison or fallback ---
# from langchain_core.runnables import RunnablePassthrough
from llm_integration.prompts import rag_prompt # Need the text-only RAG prompt
def create_rag_chain():
    """Creates the main RAG chain using LangChain Expression Language (LCEL) - TEXT ONLY."""
    try:
        # Needs logic to potentially get a *text-only* LLM instance
        # text_llm = get_text_llm() # Assumes a function exists to get non-vision model
        text_llm = get_llm() # Or reuse the main one if it handles text well
    except ValueError as e:
        log.error(f"Cannot create RAG chain: {e}")
        return None # Indicate failure

    def format_context_text_only(context_dict: dict) -> str:
        # ... (Original formatting function from text-only version) ...
        if not context_dict: return "No context available."
        # ... format title, released, tagline, directors, actors, genres ...
        return formatted_context

    rag_chain = (
        {"context": (lambda x: format_context_text_only(x["context"])), "question": RunnablePassthrough()}
        | rag_prompt # Use the text-only RAG prompt template
        | text_llm
        | StrOutputParser()
    )
    return rag_chain