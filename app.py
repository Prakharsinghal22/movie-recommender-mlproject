import streamlit as st
import pickle
import pandas as pd
import requests
import os

# ----------------------------
# TMDB API KEY (from Streamlit secrets)
# ----------------------------
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", None)

# ----------------------------
# Hugging Face file URLs
# ----------------------------
MOVIE_URL = "https://huggingface.co/datasets/Prakharsinghal22/movie-recommender-files/resolve/main/movie_dict.pkl"
SIMILARITY_URL = "https://huggingface.co/datasets/prakharsinghal22/movie-recommender-files/resolve/main/similarity.pkl"

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def download_file(url, local_path):
    """Download file if it doesn't exist locally."""
    if not os.path.exists(local_path):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            st.error(f"Error downloading {local_path}: {e}")

# Download necessary files at startup
download_file(MOVIE_URL, "movie_dict.pkl")
download_file(SIMILARITY_URL, "similarity.pkl")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    movies, similarity = None, None
    try:
        with open("movie_dict.pkl", "rb") as f:
            movies = pd.DataFrame(pickle.load(f))
    except Exception as e:
        st.error(f"Error loading movie_dict.pkl: {e}")

    try:
        with open("similarity.pkl", "rb") as f:
            similarity = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading similarity.pkl: {e}")

    return movies, similarity

movies, similarity = load_data()

# ----------------------------
# TMDB FUNCTIONS
# ----------------------------
@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    """Fetch poster and rating from TMDB."""
    if not TMDB_API_KEY:
        return None, None
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
        poster = (
            "https://image.tmdb.org/t/p/w500" + data["poster_path"]
            if data.get("poster_path") else None
        )
        rating = data.get("vote_average", 0)
        return poster, rating
    except:
        return None, None

@st.cache_data(show_spinner=False)
def fetch_trailer(movie_id):
    """Fetch YouTube trailer from TMDB."""
    if not TMDB_API_KEY:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        params = {"api_key": TMDB_API_KEY}
        data = requests.get(url, params=params, timeout=5).json()
        for video in data.get("results", []):
            if video["type"] == "Trailer" and video["site"] == "YouTube":
                return f"https://www.youtube.com/watch?v={video['key']}"
    except:
        pass
    return None

# ----------------------------
# RECOMMENDATION FUNCTION
# ----------------------------
def recommend(movie_name):
    if movies is None or similarity is None:
        st.error("Data files missing. Cannot generate recommendations.")
        return []

    if movie_name not in movies["title"].values:
        st.error("Selected movie not found in dataset.")
        return []

    movie_index = movies[movies["title"] == movie_name].index[0]
    distances = similarity[movie_index]

    ranked_movies = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    results = []
    for i in ranked_movies:
        if len(results) == 5:
            break
        row = movies.iloc[i[0]]
        poster, rating = fetch_movie_details(row.movie_id)
        trailer = fetch_trailer(row.movie_id)
        if poster:
            results.append({
                "title": row.title,
                "poster": poster,
                "rating": rating,
                "similarity": round(i[1] * 100, 2),
                "trailer": trailer
            })
    return results

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

if movies is None or similarity is None:
    st.warning("Movie data not found. Please check the files or Hugging Face links.")
else:
    selected_movie = st.selectbox("Choose a movie you like", movies["title"].values)
    if st.button("Recommend"):
        results = recommend(selected_movie)
        if results:
            cols = st.columns(len(results))
            for i, movie in enumerate(results):
                with cols[i]:
                    st.image(movie["poster"], width=200)
                    st.markdown(f"**{movie['title']}**")
                    st.markdown(f"⭐ Rating: `{movie['rating']}/10`")
                    st.markdown(f"📊 Similarity: `{movie['similarity']}%`")
                    if movie["trailer"]:
                        st.markdown(f"[▶ Watch Trailer]({movie['trailer']})", unsafe_allow_html=True)
