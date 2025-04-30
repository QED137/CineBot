from sentence_transformers import SentenceTransformer
from config import settings
import numpy as np
import logging
from typing import List

log = logging.getLogger(__name__)

_model = None

def get_embedding_model() -> SentenceTransformer:
    """Loads and returns the SentenceTransformer model."""
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        try:
            # Instantiate the SentenceTransformer model
            _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            log.info("Embedding model loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load embedding model '{settings.EMBEDDING_MODEL_NAME}': {e}")
            raise RuntimeError(f"Could not load embedding model: {e}") from e
    return _model

def generate_embedding(text: str) -> List[float]:
    """Generates a vector embedding for the given text."""
    if not text or not isinstance(text, str):
        log.warning("Attempted to generate embedding for empty or non-string input.")
        # Return a zero vector of the expected dimension or handle as error
        # Getting dimension requires loading the model first.
        try:
            model = get_embedding_model()
            dimension = model.get_sentence_embedding_dimension()
            return [0.0] * dimension
        except Exception:
             return [] # Fallback if model can't load

    try:
        model = get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist() # Convert numpy array to list for JSON/Neo4j compatibility
    except Exception as e:
        log.error(f"Error generating embedding: {e}")
        # Depending on requirements, return empty list, zero vector, or re-raise
        return []

# Example usage
if __name__ == "__main__":
    plot = "An orphaned boy enrolls in a school of wizardry, where he learns the truth about himself, his family and the terrible evil that haunts the magical world."
    embedding = generate_embedding(plot)
    if embedding:
        print(f"Generated embedding of dimension: {len(embedding)}")
        # print(embedding[:10]) # Print first 10 dimensions
    else:
        print("Failed to generate embedding.")

    empty_emb = generate_embedding("")
    print(f"Empty embedding dimension: {len(empty_emb)}")