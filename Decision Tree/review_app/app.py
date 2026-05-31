"""
app.py  --  the website
========================
Run:  python app.py
Then open http://127.0.0.1:5000 in your browser.

Needs sentiment_model.joblib in the same folder (run train_model.py first).
"""
import re
import joblib
import scipy.sparse as sp
from flask import Flask, request, jsonify, render_template_string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---- load the model saved by train_model.py ----
saved = joblib.load("sentiment_model.joblib")
vectorizer = saved["vectorizer"]
model = saved["model"]

# ---- same text cleaning the model was trained with ----
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    return " ".join(stemmer.stem(w) for w in words if w not in stop_words)


def predict_review(review, helpful_votes=0):
    text = clean_text(review)
    word_count = len(str(review).split())
    tfidf = vectorizer.transform([text])                 # transform, never fit
    X = sp.hstack([tfidf, [[helpful_votes, word_count]]])
    label = model.predict(X)[0]
    confidence = round(float(max(model.predict_proba(X)[0])), 2)
    return label, confidence


app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Sentiment Checker</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; margin: 0;
         display: flex; min-height: 100vh; align-items: center; justify-content: center; }
  .card { background: #1e293b; padding: 32px; border-radius: 16px; width: 92%;
          max-width: 560px; box-shadow: 0 20px 60px rgba(0,0,0,.4); }
  h1 { margin: 0 0 4px; font-size: 22px; }
  p.sub { margin: 0 0 20px; color: #94a3b8; font-size: 14px; }
  textarea { width: 100%; min-height: 120px; padding: 12px; border-radius: 10px;
             border: 1px solid #334155; background: #0f172a; color: #e2e8f0;
             font-size: 15px; resize: vertical; }
  button { margin-top: 14px; width: 100%; padding: 12px; border: 0; border-radius: 10px;
           background: #6366f1; color: white; font-size: 15px; font-weight: 600; cursor: pointer; }
  button:hover { background: #4f46e5; }
  #result { margin-top: 20px; padding: 16px; border-radius: 10px; text-align: center;
            font-size: 18px; font-weight: 600; display: none; }
  .pos { background: #064e3b; color: #6ee7b7; }
  .neg { background: #7f1d1d; color: #fca5a5; }
  .conf { font-size: 13px; font-weight: 400; opacity: .8; margin-top: 4px; }
</style>
</head>
<body>
  <div class="card">
    <h1>Review Sentiment Checker</h1>
    <p class="sub">Paste a product review. The decision-tree model predicts positive or negative.</p>
    <textarea id="review" placeholder="e.g. The battery life is amazing and the camera is great..."></textarea>
    <button onclick="check()">Check Sentiment</button>
    <div id="result"></div>
  </div>
<script>
async function check() {
  const text = document.getElementById('review').value.trim();
  const box = document.getElementById('result');
  if (!text) { return; }
  box.style.display = 'block';
  box.className = ''; box.textContent = 'Checking...';
  const res = await fetch('/predict', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ review: text })
  });
  const data = await res.json();
  box.className = data.label === 'positive' ? 'pos' : 'neg';
  box.innerHTML = data.label.toUpperCase() +
    '<div class="conf">confidence ' + Math.round(data.confidence * 100) + '%</div>';
}
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    review = request.get_json().get("review", "")
    label, confidence = predict_review(review)
    return jsonify({"label": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)
