"""
search.py
Provides boolean and ranked retrieval over the TF-IDF index.
"""
import pickle, json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models"

def load_index():
    with open(MODEL_PATH / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH / "doclist.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    texts = [d["title"] + " " + d["text"] for d in docs]
    doc_vectors = vectorizer.transform(texts)
    return vectorizer, docs, doc_vectors

def boolean_search(query, vectorizer, docs):
    # simple boolean: AND of terms (documents must contain all terms in either title or text)
    terms = [t.lower() for t in query.split() if t.strip()]
    results = []
    for d in docs:
        hay = (d["title"] + " " + d["text"]).lower()
        if all(t in hay for t in terms):
            results.append(d)
    return results

def ranked_search(query, vectorizer, docs, doc_vectors, topk=5):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, doc_vectors).flatten()
    idx = np.argsort(-sims)[:topk]
    results = []
    for i in idx:
        results.append({"score": float(sims[i]), "doc": docs[i]})
    return results

if __name__ == "__main__":
    vectorizer, docs, doc_vectors = load_index()
    print("Ready. Example ranked search for 'AI model medical':")
    for r in ranked_search("AI model medical", vectorizer, docs, doc_vectors):
        print(r["score"], r["doc"]["title"])
