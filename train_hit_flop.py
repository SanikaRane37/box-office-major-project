from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
REVENUE_DIR = PROJECT_DIR / "REVENUE PREDICTION"
HIT_FLOP_DIR = PROJECT_DIR / "HIT FLOP PREDICTION"
DATA_PATH = REVENUE_DIR / "boxoffice_mojo_df (1).csv"
MODEL_PATH = HIT_FLOP_DIR / "movie_sucess_classification_final.pkl"

CATEGORICAL_FEATURES = ["distributor", "rating", "main_genre", "side_genre"]
NUMERIC_FEATURES = ["widest_release", "budget"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    random_state=10,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["movie", "worldwide_gross", "budget"])
    data = data[data["budget"] > 0].copy()

    revenue_multiple = data["worldwide_gross"] / data["budget"]
    data["label"] = np.select(
        [revenue_multiple < 1, revenue_multiple < 2.5],
        [1, 0],
        default=2,
    )
    features = data[FEATURES]
    labels = data["label"]
    groups = data["movie"].astype(str).str.strip().str.casefold()

    train_indices, test_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=10).split(
            features, labels, groups
        )
    )
    validation_model = build_pipeline()
    validation_model.fit(features.iloc[train_indices], labels.iloc[train_indices])
    predictions = validation_model.predict(features.iloc[test_indices])

    print(f"Training rows: {len(train_indices)}")
    print(f"Validation rows: {len(test_indices)}")
    print(f"Validation accuracy: {accuracy_score(labels.iloc[test_indices], predictions):.3f}")
    print(
        "Validation balanced accuracy: "
        f"{balanced_accuracy_score(labels.iloc[test_indices], predictions):.3f}"
    )
    print(
        classification_report(
            labels.iloc[test_indices],
            predictions,
            labels=[0, 1, 2],
            target_names=["Average", "Flop", "Hit"],
            zero_division=0,
        )
    )

    model = build_pipeline()
    model.fit(features, labels)
    categorical_values = {
        column: sorted(data[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    model_data = {
        "model": model,
        "distributors": categorical_values["distributor"],
        "ratings": categorical_values["rating"],
        "main_genres": categorical_values["main_genre"],
        "side_genres": categorical_values["side_genre"],
        "theater_min": int(data["widest_release"].min()),
        "theater_max": int(data["widest_release"].max()),
        "theater_default": int(data["widest_release"].median()),
        "budget_min": float(data["budget"].min()),
        "budget_max": float(data["budget"].max()),
        "budget_default": float(data["budget"].median()),
    }

    HIT_FLOP_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model_data, model_file)
    print(f"Saved trained model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
