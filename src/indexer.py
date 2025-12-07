"""
indexer.py
Builds a TF-IDF vectorizer and stores document metadata for retrieval.
"""
import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_articles.json"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH.mkdir(exist_ok=True)

def build_index():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    texts = [d["title"] + " " + d["text"] for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)
    # Save
    with open(MODEL_PATH / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH / "doclist.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)
    print("Index built. Documents:", len(docs))

if __name__ == "__main__":
    build_index()
