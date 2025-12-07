"""
app.py
Simple Flask web UI for searching the sample news articles.
Run:
1) python src/indexer.py
2) python -m flask run --host=0.0.0.0 --port=5000
or: python src/app.py
"""
from flask import Flask, render_template_string, request, jsonify
from pathlib import Path
from src.search import load_index, boolean_search, ranked_search

app = Flask(__name__, static_folder="../static", template_folder="../templates")

vectorizer, docs, doc_vectors = load_index()

INDEX_HTML = """
<!doctype html>
<title>News Search Engine - CS516</title>
<h2>News Article Search Engine (CS516)</h2>
<form method="get">
  <input name="q" placeholder="Enter query" size="60" value="{{q|default('')}}"/>
  <select name="mode">
    <option value="ranked" {% if mode=='ranked' %}selected{% endif %}>Ranked (TF-IDF + Cosine)</option>
    <option value="boolean" {% if mode=='boolean' %}selected{% endif %}>Boolean (AND)</option>
  </select>
  <button type="submit">Search</button>
</form>
<hr/>
{% if q is not none %}
  <h3>Results (mode={{mode}})</h3>
  {% if results|length==0 %}<p><i>No results found.</i></p>{% endif %}
  <ol>
  {% for r in results %}
    <li>
      <strong>{{r.title}}</strong><br/>
      <small>{{r.text}}</small><br/>
      {% if r.score is defined %}<em>score: {{'%.4f'|format(r.score)}}</em>{% endif %}
    </li>
  {% endfor %}
  </ol>
{% endif %}
<hr/>
<p>Project: News Article Search Engine &middot; CS516</p>
"""

@app.route("/", methods=["GET"])
def home():
    q = request.args.get("q")
    mode = request.args.get("mode", "ranked")
    results = []
    if q:
        if mode == "boolean":
            hits = boolean_search(q, vectorizer, docs)
            results = hits
        else:
            hits = ranked_search(q, vectorizer, docs, doc_vectors, topk=10)
            results = [{"title": h["doc"]["title"], "text": h["doc"]["text"], "score": h["score"]} for h in hits]
    return render_template_string(INDEX_HTML, q=q, mode=mode, results=results)

if __name__ == "__main__":
    # allow running directly for convenience
    app.run(host="0.0.0.0", port=5000, debug=False)
