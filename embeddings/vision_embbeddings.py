#this i will use for generating vectore embedding for input picture
import torch
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging
from langchain_community.graphs import Neo4jGraph
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image_from_url(image_url: str):
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features[0].cpu().tolist()
    except Exception as e:
        logger.error(f"Image embedding failed: {e}")
        return None

def embed_uploaded_image(file) -> list:
    try:
        image = Image.open(file).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features[0].cpu().tolist()
    except Exception as e:
        logger.error(f"User image embedding failed: {e}")
        return []
