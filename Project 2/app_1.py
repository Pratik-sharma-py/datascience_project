"""
MOVIEFLIX — a deployable Netflix-style recommender frontend.

Runs the SVD collaborative-filtering model that was trained in the notebook,
using ONLY the exported numbers (recommender_data.npz) — no training library
needed, so it deploys anywhere fast.

Run locally:   streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------- page setup
st.set_page_config(page_title="MOVIEFLIX", page_icon="🎬", layout="wide")

CSS = """
<style>
  .stApp { background:#0b0b0f; color:#f5f5f7; }
  #MainMenu, footer, header { visibility:hidden; }
  .brand { font-size:30px; font-weight:800; letter-spacing:1px; color:#e50914; margin:0; }
  .brand b { color:#f5f5f7; }
  .tag { color:#9a9aa6; font-size:14px; margin-top:-6px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:14px; }
  .card { position:relative; aspect-ratio:2/3; border-radius:10px; overflow:hidden;
          border:1px solid #23232c; transition:transform .18s ease, box-shadow .18s ease; }
  .card:hover { transform:scale(1.05); box-shadow:0 10px 30px rgba(0,0,0,.6); }
  .card .meta { position:absolute; inset:0; display:flex; flex-direction:column;
                justify-content:flex-end; padding:12px; }
  .card .meta::before { content:""; position:absolute; inset:0;
                background:linear-gradient(180deg,rgba(0,0,0,.05),rgba(0,0,0,.78)); }
  .card .ttl { position:relative; z-index:1; font-weight:700; font-size:14px; line-height:1.2; }
  .card .sub { position:relative; z-index:1; color:#c9c9d2; font-size:11px; margin-top:4px; }
  .score { position:absolute; top:8px; right:8px; z-index:2; background:#e50914;
           color:#fff; font-size:11px; font-weight:700; padding:2px 7px; border-radius:6px; }
  .stSelectbox label { color:#9a9aa6 !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --- locate data files NEXT TO THIS SCRIPT (works regardless of launch folder) ---
import os
HERE = os.path.dirname(os.path.abspath(__file__))
NEEDED = ["recommender_data.npz", "movies_meta.csv", "user_seen.json"]
missing = [f for f in NEEDED if not os.path.exists(os.path.join(HERE, f))]
if missing:
    st.error("Missing data file(s): " + ", ".join(missing))
    st.info("These files must sit in the same folder as app_1.py:\n\n" + HERE +
            "\n\nGenerate them by running the export cell in your notebook, "
            "then copy all three here.")
    st.stop()

# ----------------------------------------------------------------- load data (cached)
@st.cache_resource
def load_model():
    d = np.load(os.path.join(HERE, "recommender_data.npz"), allow_pickle=True)
    model = {
        "mu": float(d["mu"]), "bu": d["bu"], "bi": d["bi"],
        "pu": d["pu"], "qi": d["qi"],
        "user_raw": d["user_raw"], "movie_raw": d["movie_raw"],
    }
    model["u_index"] = {int(u): i for i, u in enumerate(model["user_raw"])}
    model["m_index"] = {int(m): i for i, m in enumerate(model["movie_raw"])}
    model["item_sim"] = cosine_similarity(model["qi"])      # one-time at startup
    return model

@st.cache_data
def load_meta():
    meta = pd.read_csv(os.path.join(HERE, "movies_meta.csv"))
    title_of = dict(zip(meta.movieId, meta.title))
    genre_of = dict(zip(meta.movieId, meta.genres))
    seen = {int(k): set(v) for k, v in json.load(open(os.path.join(HERE, "user_seen.json"))).items()}
    return meta, title_of, genre_of, seen

M = load_model()
meta, title_of, genre_of, seen = load_meta()

# ----------------------------------------------------------------- recommendation logic
def predict(user_id, movie_id):
    est = M["mu"]
    if user_id in M["u_index"]:
        est += M["bu"][M["u_index"][user_id]]
    if movie_id in M["m_index"]:
        est += M["bi"][M["m_index"][movie_id]]
    if user_id in M["u_index"] and movie_id in M["m_index"]:
        est += float(M["pu"][M["u_index"][user_id]] @ M["qi"][M["m_index"][movie_id]])
    return float(np.clip(est, 0.5, 5.0))

def recommend_for_user(user_id, n=12):
    if user_id not in M["u_index"]:
        return []
    cand = [m for m in M["m_index"] if m not in seen.get(user_id, set())]
    scored = sorted(((m, predict(user_id, m)) for m in cand),
                    key=lambda x: x[1], reverse=True)[:n]
    return scored

def similar_movies(movie_id, n=12):
    if movie_id not in M["m_index"]:
        return []
    i = M["m_index"][movie_id]
    sims = M["item_sim"][i]
    order = [j for j in np.argsort(sims)[::-1] if j != i][:n]
    return [(int(M["movie_raw"][j]), float(sims[j])) for j in order]

# ----------------------------------------------------------------- UI helpers
def poster_style(title):
    h = 0
    for ch in str(title):
        h = (h * 31 + ord(ch)) % 360
    return f"background:linear-gradient(150deg,hsl({h},55%,30%),hsl({(h+40)%360},60%,16%));"

def render_grid(items, show_score="rating"):
    cards = []
    for movie_id, value in items:
        title = title_of.get(movie_id, str(movie_id))
        genre = str(genre_of.get(movie_id, "")).split("|")[0]
        if show_score == "rating":
            badge = f'<div class="score">★ {value:.1f}</div>'
        elif show_score == "similarity":
            badge = f'<div class="score">{value:.0%}</div>'
        else:
            badge = ""
        cards.append(
            f'<div class="card" style="{poster_style(title)}">{badge}'
            f'<div class="meta"><div class="ttl">{title}</div>'
            f'<div class="sub">{genre}</div></div></div>'
        )
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- header
st.markdown('<p class="brand">MOVIE<b>FLIX</b></p>', unsafe_allow_html=True)
st.markdown('<p class="tag">SVD collaborative-filtering recommendations</p>',
            unsafe_allow_html=True)

tab_movie, tab_user = st.tabs(["🎬  Similar to a movie", "👤  For a user"])

# ---- mode 1: pick a movie -> similar movies ----
with tab_movie:
    titles = meta.sort_values("title")["title"].tolist()
    t2id = dict(zip(meta.title, meta.movieId))
    choice = st.selectbox("Pick a movie", titles, index=0, key="movie_pick")
    movie_id = int(t2id[choice])

    g = [x for x in str(genre_of.get(movie_id, "")).split("|") if x]
    st.caption("Genres: " + ", ".join(g) if g else "")
    st.subheader("More Like This")
    st.caption("Movies that viewers with similar taste also rate highly (from the SVD factors).")
    sims = similar_movies(movie_id, 12)
    if sims:
        render_grid(sims, show_score="similarity")
    else:
        st.info("No similar titles found for this movie.")

# ---- mode 2: pick a user -> personalized recs ----
with tab_user:
    user_ids = sorted(int(u) for u in M["user_raw"])
    user_id = st.selectbox("Pick a user id", user_ids, index=0, key="user_pick")
    st.subheader(f"Top picks for user {user_id}")
    st.caption("Highest predicted ratings among movies this user hasn't rated yet.")
    recs = recommend_for_user(user_id, 12)
    if recs:
        render_grid(recs, show_score="rating")
    else:
        st.info("This user isn't in the model.")