"""
run_on_articles.py
Convenience script to build index and optionally run a simple command-line search.
"""
from src.indexer import build_index
from src.search import load_index, ranked_search, boolean_search

def main():
    build_index()
    vectorizer, docs, doc_vectors = load_index()
    print("Index ready. Try a sample ranked search for 'climate summit':")
    for r in ranked_search("climate summit", vectorizer, docs, doc_vectors):
        print(r["score"], r["doc"]["title"])

if __name__ == "__main__":
    main()
