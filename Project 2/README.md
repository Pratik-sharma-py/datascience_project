# 🎬 Movie Recommendation System

A Netflix-style movie recommender built with **collaborative filtering** and **matrix factorization** on the MovieLens dataset. It learns each user's taste from rating patterns, predicts how a user would rate movies they haven't seen, and serves recommendations through a deployable Streamlit web app.

---

## 📖 Overview

The system answers one core question:

> *"Given how many people have rated movies, what rating would a particular user give to a movie they haven't watched yet?"*

From that prediction, two recommendation features are built:

- **"For you"** — personalized top-N picks for a given user.
- **"More like this"** — movies similar to a selected movie (item-to-item).

Both are powered by a single trained matrix-factorization model.

---

## What this project does

- **EDA** (`notebooks/eda.ipynb`) — explores the dataset: sparsity, rating distribution, the popularity long-tail, genres, rater behaviour, and release-year trends.
- **Modeling pipeline** (`notebooks/netflix_recommendation.ipynb`) — loads and cleans the data, compares five algorithms on the same test set, trains a final matrix-factorization model on all data, and builds both recommendation features.
- **Recommendations** — `recommend_for_user()` (personalized) and `similar_movies()` (movie-to-movie, from the learned movie factors).
- **Deployment** — a Streamlit app (`app_1.py`) that serves recommendations using exported model artifacts; a `Dockerfile` is included for containerized deployment.
- **Testing** — a preprocessing test in `tests/test_preprocess.py`.

---

## 📊 Dataset

[MovieLens](https://grouplens.org/datasets/movielens/) — the standard benchmark dataset for recommender systems, from the GroupLens research group.

| File | Columns | Description |
|------|---------|-------------|
| `data/raw/rating.csv` | `userId`, `movieId`, `rating`, `timestamp` | User ratings on a 0.5–5.0 scale |
| `data/raw/movie.csv`  | `movieId`, `title`, `genres` | Movie catalog with pipe-separated genres |

> The `timestamp` column is skipped during loading to save memory.

---

## 🧠 Methodology

### Collaborative filtering
The system uses **collaborative filtering** — it learns purely from rating patterns ("users who rated these movies this way also rated those movies that way"), without needing to understand a movie's content.

### Matrix factorization (the final model)
The final model is **matrix factorization** — the method made famous by the Netflix Prize. Every user and every movie is represented by a short vector of hidden **latent factors** ("taste dials"). A predicted rating is the dot product of a user's factors with a movie's factors, plus bias terms:

```
predicted_rating ≈ global_average + user_bias + movie_bias + (user_factors · movie_factors)
```

The model learns these factors from the data; it is never told what they mean.

### Algorithms compared
Five algorithms are evaluated on the same held-out test set:

| Family | Algorithm | Idea |
|--------|-----------|------|
| Baseline | `NormalPredictor` | Random guess — the bar every model must beat |
| Neighborhood | `KNN` (item-based) | "This movie is similar to others you liked" |
| Neighborhood | `KNN` (user-based) | "People similar to you liked it" |
| Matrix factorization | `SVD` | Hidden taste factors (Funk SVD) |
| Matrix factorization | `NMF` | Hidden factors, constrained to be non-negative |

### Evaluation metrics
- **RMSE** (Root Mean Squared Error) — average prediction error, penalizing large misses more. *Lower is better.* (The official Netflix Prize metric.)
- **MAE** (Mean Absolute Error) — average error in plain rating points.

Models train on 80% of the ratings and are scored on the unseen 20%, so scores reflect performance on data the model has never seen.

---

##  Results

The comparison produces a leaderboard. **Replace these with your own run's numbers** — exact values depend on the sample size and filtering settings.

| Algorithm | Family | RMSE | MAE |
|-----------|--------|------|-----|
| SVD | Matrix factorization | _your value_ | _your value_ |
| NMF | Matrix factorization | _your value_ | _your value_ |
| KNN user-based | Neighborhood | _your value_ | _your value_ |
| KNN item-based | Neighborhood | _your value_ | _your value_ |
| Random baseline | Baseline | _your value_ | _your value_ |

The matrix-factorization models lead; the random baseline is clearly worst, confirming the models learned real signal. The best matrix-factorization model is retrained on all data and used for serving.

---

## 📁 Project structure

```
Project 2/
├── data/
│   ├── raw/                          # original data
│   │   ├── movie.csv
│   │   └── rating.csv
│   ├── interim/                      # intermediate data (placeholder)
│   └── processed/                    # cleaned data (placeholder)
├── notebooks/
│   ├── eda.ipynb                     # exploratory data analysis
│   └── netflix_recommendation.ipynb  # main pipeline: clean → compare → train → recommend → export
├── src/                              # source modules (placeholder)
├── tests/
│ 
├── models/                           # saved models (placeholder)
├── logs/                             # logs (placeholder)
├── reports/
│   └── figures/                      # generated figures (placeholder)
├── app_1.py                          # Streamlit web app
├── recommender_data.npz              # exported model factors (used by the app)
├── movies_meta.csv                   # movieId, title, genres (used by the app)
├── user_seen.json                    # movies each user already rated (used by the app)
├── Dockerfile                        # containerized deployment
├── Makefile                          # project commands
├── pyproject.toml                    # project metadata / dependencies
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

> The layout follows the standard cookiecutter-data-science structure. Folders marked *(placeholder)* are part of that layout and can be filled in as the project grows.

---

## ⚙️ Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Key libraries: `scikit-surprise`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`.

---

## 🚀 How to run

### 1. Explore the data
Open `notebooks/eda.ipynb` and run it to understand the dataset.

### 2. Train the model
Open `notebooks/netflix_recommendation.ipynb`, set the data path, and **Run All**. The notebook will:
1. Load and clean the data.
2. Compare the five algorithms and show a leaderboard.
3. Train the final matrix-factorization model on all data.
4. Demonstrate `recommend_for_user()` and `similar_movies()`.
5. Export the model artifacts.

> Tip: keep `DEV_MODE = True` (samples 10,000 users) while iterating, then set it to `False` for the full dataset.

### 3. Export the model for the app
The export cell at the end of the notebook writes three files that the app needs:
`recommender_data.npz`, `movies_meta.csv`, `user_seen.json`. These live in the project root, next to `app_1.py`.

### 4. Run the web app
```bash
streamlit run app_1.py
```

The app opens at `http://localhost:8501` with two tabs: **Similar to a movie** and **For a user**.

---

## Deployment

The app loads only the exported model numbers (a few small matrices) and reconstructs predictions in NumPy, so it needs no training library and is light enough to host for free.

**Streamlit Community Cloud:**
1. Push the repo (including `app_1.py`, `requirements.txt`, and the three data files) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, select the repo, set the main file to `app_1.py`, and **Deploy**.

**Docker (optional):** a `Dockerfile` is included for containerized deployment.

```bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
```

---

##  How it works (under the hood)

1. **Load & clean** — read ratings with small dtypes; filter out users/movies with too few ratings.
2. **Split** — hold out 20% of ratings as a test set for honest evaluation.
3. **Compare** — train baseline, KNN, SVD, and NMF; score each on the test set.
4. **Train final** — retrain the best matrix-factorization model on *all* ratings.
5. **Recommend** —
   - *Per-user:* predict every unseen movie, return the highest.
   - *Per-movie:* cosine similarity between movies' learned factor vectors for "more like this".
6. **Export & serve** — save the model's factor matrices and serve predictions in the Streamlit app.

---

##  Limitations

- **Cold start** — cannot recommend for brand-new users or movies with no ratings yet.
- **Popularity bias** — like most collaborative filtering, it tends to favor popular titles.
- **No content understanding** — the core model knows only who rated a movie and how, not what the movie is about.

---

## 🔮 Future work

- Add **content features** (genres, release year) for a **hybrid** model to ease cold-start.
- Try **SVD++** for a small accuracy gain.
- Tune hyperparameters with `GridSearchCV`.
- Add **ranking metrics** (Precision@K, Recall@K, NDCG) alongside RMSE/MAE.
- Fetch real movie posters via the TMDB API for the web app.
- Explore **deep-learning recommenders** (neural collaborative filtering, two-tower models).

---

## 🛠️ Tech stack

`Python` · `pandas` · `NumPy` · `scikit-surprise` · `scikit-learn` · `matplotlib` · `Streamlit` · `Docker`

##  Acknowledgments

- [MovieLens / GroupLens](https://grouplens.org/datasets/movielens/) for the dataset.
- The [Surprise](https://surpriselib.com/) library for the recommendation algorithms.
