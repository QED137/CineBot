from PIL import Image
import io
import logging

log = logging.getLogger(__name__)

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Loads an image from bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary (CLIP models often expect RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        log.error(f"Error loading image from bytes: {e}")
        raise ValueError("Could not load image data") from e

# Add other functions if needed (e.g., resizing, normalization specific to model)
# Many transformers processors handle resizing/normalization internally.