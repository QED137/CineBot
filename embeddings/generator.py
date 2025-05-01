# from sentence_transformers import SentenceTransformer
# from config import settings
# import numpy as np
# import logging
# from typing import List

# log = logging.getLogger(__name__)

# _model = None

# def get_embedding_model() -> SentenceTransformer:
#     """Loads and returns the SentenceTransformer model."""
#     global _model
#     if _model is None:
#         log.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
#         try:
#             # Instantiate the SentenceTransformer model
#             _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
#             log.info("Embedding model loaded successfully.")
#         except Exception as e:
#             log.error(f"Failed to load embedding model '{settings.EMBEDDING_MODEL_NAME}': {e}")
#             raise RuntimeError(f"Could not load embedding model: {e}") from e
#     return _model

# def generate_embedding(text: str) -> List[float]:
#     """Generates a vector embedding for the given text."""
#     if not text or not isinstance(text, str):
#         log.warning("Attempted to generate embedding for empty or non-string input.")
#         # Return a zero vector of the expected dimension or handle as error
#         # Getting dimension requires loading the model first.
#         try:
#             model = get_embedding_model()
#             dimension = model.get_sentence_embedding_dimension()
#             return [0.0] * dimension
#         except Exception:
#              return [] # Fallback if model can't load

#     try:
#         model = get_embedding_model()
#         embedding = model.encode(text, convert_to_numpy=True)
#         return embedding.tolist() # Convert numpy array to list for JSON/Neo4j compatibility
#     except Exception as e:
#         log.error(f"Error generating embedding: {e}")
#         # Depending on requirements, return empty list, zero vector, or re-raise
#         return []

# # Example usage
# if __name__ == "__main__":
#     plot = "An orphaned boy enrolls in a school of wizardry, where he learns the truth about himself, his family and the terrible evil that haunts the magical world."
#     embedding = generate_embedding(plot)
#     if embedding:
#         print(f"Generated embedding of dimension: {len(embedding)}")
#         # print(embedding[:10]) # Print first 10 dimensions
#     else:
#         print("Failed to generate embedding.")

#     empty_emb = generate_embedding("")
#     print(f"Empty embedding dimension: {len(empty_emb)}")
from sentence_transformers import SentenceTransformer
from config import settings
import numpy as np
import logging
from typing import List, Optional
import requests # To fetch image for embedding during load
from PIL import Image
import io

# Keep existing text embedding functions
from multimodal.vision_model import get_clip_image_embedding # Import image embedder
from multimodal.image_processor import load_image_from_bytes

log = logging.getLogger(__name__)

_text_model = None
# Poster base URL for TMDb
POSTER_BASE_URL_LOAD = "https://image.tmdb.org/t/p/w185" # Use a consistent size

def get_text_embedding_model() -> SentenceTransformer:
    """Loads and returns the SentenceTransformer model."""
    global _text_model
    if _text_model is None:
        log.info(f"Loading text embedding model: {settings.TEXT_EMBEDDING_MODEL_NAME}")
        try:
            _text_model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL_NAME)
            log.info("Text embedding model loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load text embedding model '{settings.TEXT_EMBEDDING_MODEL_NAME}': {e}")
            raise RuntimeError(f"Could not load text embedding model: {e}") from e
    return _text_model

def generate_text_embedding(text: str) -> List[float]:
    """Generates a vector embedding for the given text."""
    if not text or not isinstance(text, str):
        log.warning("Attempted to generate text embedding for empty or non-string input.")
        try:
            model = get_text_embedding_model()
            dimension = model.get_sentence_embedding_dimension()
            return [0.0] * dimension
        except Exception:
             return []

    try:
        model = get_text_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        log.error(f"Error generating text embedding: {e}")
        return []

# --- NEW: Function to generate image embedding during data load ---
def generate_image_embedding_from_url(poster_path: Optional[str]) -> List[float]:
    """
    Fetches poster image from TMDb url (using poster_path) and generates embedding.
    Used during the data loading script.
    """
    if not poster_path:
        log.debug("No poster path provided, skipping image embedding generation.")
        return []

    image_url = f"{POSTER_BASE_URL_LOAD}{poster_path}"
    log.debug(f"Fetching image for embedding from: {image_url}")

    try:
        response = requests.get(image_url, timeout=10) # Add timeout
        response.raise_for_status() # Check for HTTP errors

        image_bytes = response.content
        image = load_image_from_bytes(image_bytes)

        # Generate embedding using the vision model function
        embedding = get_clip_image_embedding(image)
        if embedding:
             log.debug(f"Successfully generated embedding for poster: {poster_path}")
        else:
             log.warning(f"Failed to generate embedding for poster: {poster_path}")
        return embedding

    except requests.exceptions.RequestException as e:
        log.warning(f"Failed to fetch image from {image_url}: {e}")
        return []
    except ValueError as e: # Catch image loading errors
        log.warning(f"Failed to load image from {image_url}: {e}")
        return []
    except Exception as e:
        log.error(f"Unexpected error generating image embedding for {poster_path}: {e}")
        return []