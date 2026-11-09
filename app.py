from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_DIR = Path(__file__).resolve().parent
RECOMMENDATION_DIR = PROJECT_DIR / "RECOMENDATION SYSTEM"
HIT_FLOP_DIR = PROJECT_DIR / "HIT FLOP PREDICTION"
SENTIMENT_DIR = PROJECT_DIR / "SENTIMENT ANALYSIS"
REVENUE_DIR = PROJECT_DIR / "REVENUE PREDICTION"

st.set_page_config(page_title="Box Office Studio", page_icon="🎬", layout="wide")


@st.cache_resource
def load_pickle(path: str):
    with open(path, "rb") as model_file:
        return pickle.load(model_file)


def show_missing_artifact(path: Path) -> None:
    st.error(f"Required model file is missing: `{path.name}`")
    st.caption(f"Place the model file in this folder: `{path.parent}`")


@st.cache_resource
def load_recommendation_data():
    csv_path = RECOMMENDATION_DIR / "movies_infinity.csv"
    movies = pd.read_csv(csv_path)
    required_columns = [
        "Title",
        "Title_URL",
        "poster_img",
        "tagline",
        "overview",
        "keywords",
        "cast1",
        "cast2",
        "cast3",
        "genres",
    ]
    missing_columns = [column for column in required_columns if column not in movies]
    if missing_columns:
        raise ValueError(f"Recommendation data is missing columns: {', '.join(missing_columns)}")

    text_columns = ["tagline", "overview", "keywords", "cast1", "cast2", "cast3", "genres"]
    text = movies[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    movie_vectors = vectorizer.fit_transform(text)
    return movies, movie_vectors


def show_recommendations() -> None:
    st.header("Movie recommendations")
    try:
        movies, movie_vectors = load_recommendation_data()
    except (OSError, ValueError) as error:
        st.error(f"Could not load recommendation data: {error}")
        return

    titles = movies["Title"].dropna().astype(str)
    titles = titles[titles.str.strip().ne("")].drop_duplicates().tolist()
    if not titles:
        st.error("The recommendation dataset has no movie titles.")
        return

    selected_title = st.selectbox("Choose a movie", titles)
    if not st.button("Recommend movies", key="recommend_movies"):
        return

    selected_index = movies.index[movies["Title"].astype(str) == selected_title][0]
    similarities = movie_vectors.getrow(selected_index).dot(movie_vectors.T).toarray().ravel()
    similarities[selected_index] = -1
    result_indices = np.argsort(similarities)[::-1][:5]
    if not len(result_indices):
        st.warning("No similar movies were found.")
        return

    st.subheader(f"Movies similar to {selected_title}")
    columns = st.columns(min(5, len(result_indices)))
    for column, movie_index in zip(columns, result_indices):
        movie = movies.iloc[movie_index]
        with column:
            poster = movie["poster_img"]
            if pd.notna(poster) and str(poster).strip():
                st.image(str(poster), use_column_width=True)
            st.markdown(f"**{movie['Title']}**")
            tagline = movie["tagline"]
            if pd.notna(tagline) and str(tagline).strip():
                st.caption(str(tagline))
            movie_url = movie["Title_URL"]
            if pd.notna(movie_url) and str(movie_url).strip():
                st.markdown(f"[Movie details]({movie_url})")


def show_hit_flop() -> None:
    st.header("Hit / flop prediction")
    st.caption(
        "Experimental estimate trained on the project's box-office data. "
        "Labels are based on worldwide gross divided by budget: below 1x is flop, "
        "1x–2.5x is average, and 2.5x or more is hit."
    )
    classifier_path = HIT_FLOP_DIR / "movie_sucess_classification_final.pkl"
    if not classifier_path.is_file():
        show_missing_artifact(classifier_path)
        st.info("Run `python train_hit_flop.py` from the project folder to create this model.")
        return

    model_data = load_pickle(str(classifier_path))
    required_keys = {
        "model",
        "distributors",
        "ratings",
        "main_genres",
        "side_genres",
        "theater_min",
        "theater_max",
        "theater_default",
        "budget_min",
        "budget_max",
        "budget_default",
    }
    missing_keys = required_keys.difference(model_data)
    if missing_keys:
        st.error(f"The hit/flop model is missing required data: {', '.join(sorted(missing_keys))}")
        return

    distributor = st.selectbox("Distributor", model_data["distributors"])
    rating = st.selectbox("Rating", model_data["ratings"])
    main_genre = st.selectbox("Main genre", model_data["main_genres"])
    side_genre = st.selectbox("Side genre", model_data["side_genres"])
    theaters = st.number_input(
        "Widest release theaters",
        min_value=int(model_data["theater_min"]),
        max_value=int(model_data["theater_max"]),
        value=int(model_data["theater_default"]),
    )
    budget = st.number_input(
        "Budget (USD millions)",
        min_value=float(model_data["budget_min"]),
        max_value=float(model_data["budget_max"]),
        value=float(model_data["budget_default"]),
    )

    if st.button("Predict verdict", key="predict_verdict"):
        features = pd.DataFrame(
            [
                {
                    "distributor": distributor,
                    "rating": rating,
                    "widest_release": theaters,
                    "budget": budget,
                    "main_genre": main_genre,
                    "side_genre": side_genre,
                }
            ]
        )
        prediction = model_data["model"].predict(features)[0]
        verdicts = {0: "AVERAGE MOVIE", 1: "FLOP MOVIE", 2: "HIT MOVIE"}
        verdict = verdicts.get(prediction)
        if verdict is None:
            st.error(f"The classifier returned an unsupported verdict: {prediction}")
            return
        st.success(f"Predicted verdict: {verdict}.")


def predict_sentiment(text: str, vectorizer, classifier) -> str:
    prediction = classifier.predict(vectorizer.transform([text]))[0]
    return "Positive" if prediction == 1 else "Negative"


def split_reviews(contents: str) -> list[str]:
    numbered_reviews = re.split(r"(?m)^\s*\d+\.\s*", contents)
    reviews = [review.strip() for review in numbered_reviews if review.strip()]
    return reviews or ([contents.strip()] if contents.strip() else [])


def show_sentiment() -> None:
    st.header("Review sentiment")
    vectorizer_path = SENTIMENT_DIR / "count-vectorizer.pkl"
    classifier_path = SENTIMENT_DIR / "movie_review_classification.pkl"
    missing = [path for path in (vectorizer_path, classifier_path) if not path.is_file()]
    if missing:
        for path in missing:
            show_missing_artifact(path)
        return

    vectorizer = load_pickle(str(vectorizer_path))
    classifier = load_pickle(str(classifier_path))

    review = st.text_area("Enter a movie review")
    if st.button("Analyze review", key="analyze_review"):
        if not review.strip():
            st.warning("Enter a review before analyzing.")
        else:
            sentiment = predict_sentiment(review, vectorizer, classifier)
            st.success(f"Sentiment: {sentiment}")

    uploaded_file = st.file_uploader("Or upload a numbered .txt file", type=["txt"])
    if uploaded_file is not None and st.button("Analyze uploaded reviews", key="analyze_upload"):
        try:
            contents = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("The uploaded file must be UTF-8 encoded.")
            return

        reviews = split_reviews(contents)
        if not reviews:
            st.warning("The uploaded file contains no review text.")
            return

        results = [predict_sentiment(item, vectorizer, classifier) for item in reviews]
        st.write(f"Positive reviews: {results.count('Positive')}")
        st.write(f"Negative reviews: {results.count('Negative')}")
        st.dataframe(
            pd.DataFrame({"Review": reviews, "Sentiment": results}),
            use_container_width=True,
        )


REVENUE_MODELS = {
    "Opening revenue": "saved_steps_opening.pkl",
    "Domestic revenue": "saved_steps_dom.pkl",
    "Worldwide revenue": "saved_steps_worldwide.pkl",
}


def encoder_values(encoder) -> list:
    values = encoder.categories_[0].tolist()
    return [value for value in values if not pd.isna(value)]


def show_revenue() -> None:
    st.header("Box-office revenue prediction")
    revenue_type = st.selectbox("Revenue type", list(REVENUE_MODELS))
    model_path = REVENUE_DIR / REVENUE_MODELS[revenue_type]
    if not model_path.is_file():
        show_missing_artifact(model_path)
        return

    model_data = load_pickle(str(model_path))
    required_keys = {"model", "oe_distributor", "oe_rating", "oe_mgenre", "oe_sgenre"}
    missing_keys = required_keys.difference(model_data)
    if missing_keys:
        st.error(f"The model file is missing required data: {', '.join(sorted(missing_keys))}")
        return

    distributors = encoder_values(model_data["oe_distributor"])
    ratings = encoder_values(model_data["oe_rating"])
    main_genres = encoder_values(model_data["oe_mgenre"])
    side_genres = encoder_values(model_data["oe_sgenre"])
    if not all((distributors, ratings, main_genres, side_genres)):
        st.error("The selected model does not contain usable input categories.")
        return

    distributor = st.selectbox("Distributor", distributors)
    theaters = st.number_input("Opening theaters", min_value=1, max_value=5000, value=1000)
    rating = st.selectbox("Rating", ratings)
    main_genre = st.selectbox("Main genre", main_genres)
    side_genre = st.selectbox("Side genre", side_genres)
    budget = st.number_input("Budget", min_value=1, max_value=500, value=50)

    if st.button("Predict revenue", key="predict_revenue"):
        features = np.array(
            [[distributor, rating, theaters, budget, main_genre, side_genre]],
            dtype=object,
        )
        encoders = {
            0: model_data["oe_distributor"],
            1: model_data["oe_rating"],
            4: model_data["oe_mgenre"],
            5: model_data["oe_sgenre"],
        }
        for column, encoder in encoders.items():
            features[:, column] = encoder.transform(features[:, column].reshape(-1, 1)).ravel()

        try:
            prediction = model_data["model"].predict(features.astype(float))[0]
        except AttributeError as error:
            if "monotonic_cst" not in str(error):
                raise
            st.error(
                "This saved model is incompatible with the installed scikit-learn version. "
                "Install the project's pinned dependencies and restart the app."
            )
            return
        st.metric("Predicted revenue (model output)", f"{prediction:,.2f}")


st.title("Box Office Studio")
st.caption("Movie recommendations, verdict predictions, review sentiment, and box-office revenue.")
page = st.sidebar.radio(
    "Navigate",
    (
        "Recommendations",
        "Hit / flop verdict",
        "Sentiment analysis",
        "Revenue prediction",
    ),
)

if page == "Recommendations":
    show_recommendations()
elif page == "Hit / flop verdict":
    show_hit_flop()
elif page == "Sentiment analysis":
    show_sentiment()
else:
    show_revenue()
