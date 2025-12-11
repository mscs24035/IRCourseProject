# News Article Search Engine - CS516 (Coding Demo)

**Title:** News Article Search Engine Using Information Retrieval Techniques
**Course:** CS 516 - Information Retrieval and Text Mining (ITU Fall 2025)
**Author:** Raheel Riaz (Individual Project)

## What this project includes
- Simple TF-IDF based inverted/semantic index using `scikit-learn`
- Boolean (AND) search and Ranked retrieval (TF-IDF + Cosine similarity)
- Minimal Flask web UI for queries and result viewing
- Sample dataset (`data/sample_articles.json`) with 5 news articles

## How to run (locally)
1. Create virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Build the index:
   ```bash
   python src/indexer.py
   ```
3. Run the Flask app:
   ```bash
   python src/app.py
   # or: FLASK_APP=src.app flask run --host=0.0.0.0 --port=5000
   ```
4. Open http://127.0.0.1:5000 in your browser and try queries like `climate summit`, `AI model`, or `local elections`.

## Files included
- `src/` - source code (indexer, search, app)
- `data/sample_articles.json` - sample dataset
- `models/` - where built vectorizer & doclist are saved after running indexer
- `requirements.txt` - Python dependencies
- `slides.pdf` - short slides for presentation
- `README.md` - this file

## Notes for submission
- This is a minimal working demo as required by the project deliverables. It demonstrates preprocessing (stop-word removal via TfidfVectorizer), index construction, Boolean and Ranked retrieval, and a UI.
- For a larger dataset (BBC/Kaggle), replace `data/sample_articles.json` with your dataset and run `src/indexer.py`.
