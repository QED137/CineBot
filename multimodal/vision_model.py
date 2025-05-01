from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import logging
from typing import List
from config import settings

log = logging.getLogger(__name__)

_vision_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {_device}")

def _get_vision_model_and_processor():
    """Loads the CLIP model and processor."""
    global _vision_model, _processor
    if _vision_model is None or _processor is None:
        log.info(f"Loading vision model: {settings.VISION_EMBEDDING_MODEL_NAME}")
        try:
            _vision_model = CLIPModel.from_pretrained(settings.VISION_EMBEDDING_MODEL_NAME).to(_device)
            _processor = CLIPProcessor.from_pretrained(settings.VISION_EMBEDDING_MODEL_NAME)
            log.info("Vision model and processor loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load vision model '{settings.VISION_EMBEDDING_MODEL_NAME}': {e}")
            raise RuntimeError(f"Could not load vision model: {e}") from e
    return _vision_model, _processor

def get_clip_image_embedding(image: Image.Image) -> List[float]:
    """Generates a CLIP vector embedding for the given PIL image."""
    if not image:
        log.warning("Attempted to generate embedding for empty image input.")
        return [] # Or return zero vector of appropriate dimension

    try:
        model, processor = _get_vision_model_and_processor()
        # Process the image (handles resizing, normalization) and move to device
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        # Generate the embedding (no gradient calculation needed)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Move features to CPU, convert to numpy, then list
        embedding = image_features.cpu().numpy().flatten().tolist()
        log.debug(f"Generated image embedding of dimension: {len(embedding)}")
        return embedding
    except Exception as e:
        log.error(f"Error generating image embedding: {e}")
        return []

# Example usage (optional)
if __name__ == "__main__":
    try:
        # Create a dummy black image for testing
        dummy_image = Image.new('RGB', (224, 224), color = 'black')
        embedding = get_clip_image_embedding(dummy_image)
        if embedding:
            print(f"Generated dummy image embedding of dimension: {len(embedding)}")
        else:
            print("Failed to generate dummy image embedding.")
    except Exception as e:
        print(f"Error in example usage: {e}")