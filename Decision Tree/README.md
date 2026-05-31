# iPhone XR Review Sentiment — Decision Tree

Predicts whether an Amazon product review is **positive** or **negative**
using a Decision Tree trained on TF-IDF features. Includes a small Flask
web app to check any review.

## Project structure
```
Decision Tree/
├── data/
│   └── raw/
│       └── apple_iphone_11_reviews.csv   # source data (iPhone XR reviews)
├── notebooks/
│   └── amazon_main.ipynb                 # EDA + training pipeline
├── review_app/
│   ├── app.py                            # Flask web app
│   └── sentiment_model.joblib            # trained model (saved from notebook)
├── requirements.txt
└── README.md
```

## Setup
Install the dependencies (once):
```
pip install -r requirements.txt
```

## Train the model
The model is trained in `notebooks/amazon_main.ipynb`. Running the notebook
end to end produces `sentiment_model.joblib` (saved via `joblib.dump`).

## Run the web app
```
cd review_app
python app.py
```
Then open http://127.0.0.1:5000, paste a review, and click **Check Sentiment**.

## How it works
1. **Clean** the text — lowercase, strip symbols, remove stopwords, stem.
2. **Vectorize** with TF-IDF (unigrams + bigrams) and add two numeric
   features (helpful votes, word count).
3. **Classify** with a depth-limited, class-balanced Decision Tree.

## Model performance
- Cross-validated macro-F1: ~0.68 (vs. ~0.47 for an always-positive baseline)
- Negative-class recall: ~0.80 (catches most complaints)

## Limitations
- Binary only (positive vs negative); neutral 3-star reviews are excluded.
- Trained on iPhone XR / Amazon India reviews (2018-2020); best on similar data.
