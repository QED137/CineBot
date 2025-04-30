# General utility functions can go here.
# For example, text cleaning, data formatting, etc.

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not text:
        return ""
    # Add more cleaning steps as needed (e.g., removing punctuation, lowercasing)
    return text.strip()
