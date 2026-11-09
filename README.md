# Box Office Major Project

This project has one Streamlit app with sidebar navigation for movie recommendations, hit/flop verdicts, sentiment analysis, and revenue predictions.

## Run the app

Open PowerShell in this project folder and run:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser and choose a page from **Navigate** in the sidebar.

For an isolated environment, run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Publish this project to GitHub

The local virtual environment and Python cache files are excluded by `.gitignore`. Project datasets and app model artifacts are retained. The 176 MB `pkl_files/similarity.pkl` backup is excluded because GitHub rejects files over 100 MB and the unified app does not load it; the original local file is not deleted. Create a **private** empty repository named `box-office-major-project` on GitHub (do not initialize it with a README, license, or `.gitignore`), then run these commands from this project folder, replacing `<YOUR-USERNAME>` with your GitHub username:

```powershell
git remote add origin https://github.com/<YOUR-USERNAME>/box-office-major-project.git
git push -u origin main
```

Git may ask you to authenticate using your configured Git Credential Manager or browser sign-in. Review the files and repository visibility on GitHub after pushing. Do not put access tokens or passwords in the remote URL or commit them to the project.

## Check each section

- **Recommendations:** choose a movie and click **Recommend movies**. Similarity is calculated from the movie text fields in `RECOMENDATION SYSTEM/movies_infinity.csv`. Poster images need internet access.
- **Hit / flop verdict:** choose the distributor, rating, genres, widest-release theater count, and budget, then click **Predict verdict**. See [Hit/flop classifier: data, training, and limitations](#hitflop-classifier-data-training-and-limitations) below for the missing-original-model explanation and methodology.
- **Sentiment analysis:** enter a review such as `I really enjoyed this movie` and click **Analyze review**. You can also upload a UTF-8 `.txt` file with numbered reviews. Both sentiment pickle files are included.
- **Revenue prediction:** choose opening, domestic, or worldwide revenue, fill in the movie attributes, and click **Predict revenue**. The corresponding saved model file must be present in `REVENUE PREDICTION`.

The app loads model files only when their page is selected. If an artifact is missing or incompatible, check the message shown on that page. The saved models were created with scikit-learn 1.3.0, so `requirements.txt` pins the compatible 1.3 release line. Pickle files should only be used if they come from a trusted source.

## Hit/flop classifier: data, training, and limitations

### Why the original classifier was unavailable

The original hit/flop notebook refers to `movie_dataset.csv`, `movie_credit.csv`, and `movie.csv`, which are not included in this project. The original classifier file, `movie_sucess_classification_final.pkl`, was also absent. The project did include a saved CountVectorizer, but a vectorizer alone cannot make predictions: it must be paired with the classifier trained on compatible features and labels. The sentiment and recommendation pickle files are different models and are not substitutes.

To keep the hit/flop page usable, this project includes a **replacement experimental classifier** trained by `train_hit_flop.py` from the available `REVENUE PREDICTION/boxoffice_mojo_df (1).csv` data. It is not the missing original model and does not reproduce the original notebook's target labels.

### Target definition and input features

The source dataset does not provide an explicit hit/flop target. The training script creates a proxy label from `worldwide_gross / budget`:

| Class | Label | Rule |
|---|---:|---|
| Flop | 1 | Worldwide gross / budget < 1 |
| Average | 0 | 1 <= worldwide gross / budget < 2.5 |
| Hit | 2 | Worldwide gross / budget >= 2.5 |

These are project-specific heuristic thresholds, not a standardized film-industry definition. The model uses distributor, rating, main genre, side genre, widest release theater count, and budget. Budget and gross values use the units present in the source dataset (budget is presented in the app as USD millions). Categorical features are imputed and one-hot encoded; numeric features are median-imputed and standardized. A balanced Random Forest with 500 trees is used.

### Validation results

The training script reserves a 20% validation split using `GroupShuffleSplit`, grouping rows by normalized movie title so duplicate titles are not split between training and validation. With the project's current dataset and random seed 10, the saved run produced:

- Training rows: 2,089
- Validation rows: 531
- Accuracy: 56.5%
- Balanced accuracy: 55.7%
- Average: precision 0.45, recall 0.34, F1-score 0.39 (163 validation examples)
- Flop: precision 0.56, recall 0.67, F1-score 0.61 (138 validation examples)
- Hit: precision 0.63, recall 0.66, F1-score 0.65 (230 validation examples)

These figures describe one held-out split of this dataset, not an independent external evaluation. Re-running training prints the validation results and then fits the saved pipeline on all eligible rows.

### Important limitations for interpretation

- Worldwide box-office gross is not studio profit. Gross does not account for theaters' share, marketing, distribution costs, financing, or other revenue streams, and the budget field may not capture total production costs.
- The target classes are derived from budget and gross, not manually verified success labels. A film can therefore receive a label that differs from common critical, commercial, or industry usage.
- The model is trained on the available historical dataset. Its performance may not generalize to other time periods, markets, distributors, or movies with input values outside the training data.
- The reported validation score is modest and comes from a single split; no claim of production-grade accuracy or causal explanation is made.

For an academic report, describe this component as an **experimental proxy-label classifier**, document the threshold choices, and distinguish its results from those of the unavailable original model.

### Retraining the replacement model

From the project folder, install the project dependencies if needed and run:

```powershell
python train_hit_flop.py
```

The script writes `HIT FLOP PREDICTION/movie_sucess_classification_final.pkl`, which the unified app loads for the hit/flop page. The file is generated locally and should be kept with the project when moving or submitting the app.

## Syntax check

From the project folder, compile-check the Python files:

```powershell
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
```

## Project layout

```text
app.py                  Unified Streamlit app
HIT FLOP PREDICTION/    Hit/flop model and notebook
RECOMENDATION SYSTEM/   Recommendation data, model, and notebook
REVENUE PREDICTION/     Revenue models, pages, and notebooks
SENTIMENT ANALYSIS/     Sentiment model, reviews, and notebook
```
