from transformers import pipeline
from PIL import Image
import io

# Only load once to avoid reloading on every request
_clip_classifier = None

def is_valid_movie_poster(image_bytes: bytes, threshold: float = 0.75) -> bool:
    global _clip_classifier
    if _clip_classifier is None:
        _clip_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    candidate_labels = ["a movie poster", "a selfie", "a human face", "a landscape", "an object"]

    result = _clip_classifier(image, candidate_labels=candidate_labels)
    top_label = result[0]["label"]
    top_score = result[0]["score"]

    # Logging for debugging
    print(f"[Poster Check] Top Label: {top_label}, Score: {top_score:.2f}")

    return top_label == "a movie poster" and top_score >= threshold
