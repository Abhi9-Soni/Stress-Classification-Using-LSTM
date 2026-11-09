"""Shared deterministic preprocessing for training and inference."""
import re
import unicodedata

def clean_text(text):
    """Deterministic normalization shared by training and inference."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " email ", text)
    text = re.sub(r"(?<!\w)(?:u/|@)[\w-]+", " user ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()
