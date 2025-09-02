# miniproject-4
# This cell creates a GitHub-ready project folder with code, README, and a Streamlit app.
import os, textwrap, json, zipfile, pathlib, sys

BASE = "/mnt/data/ai-echo-sentiment"
os.makedirs(BASE, exist_ok=True)
dirs = [
    "src",
    "app",
    "models",
    "reports",
    "data",
    "notebooks",
    ".github/workflows"
]
for d in dirs:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)

# ---------- requirements.txt ----------
requirements = textwrap.dedent("""\
    # Core
    pandas==2.2.2
    numpy==1.26.4
    scikit-learn==1.5.1
    joblib==1.4.2
    
    # NLP
    nltk==3.9.1
    langdetect==1.0.9
    
    # Viz
    matplotlib==3.9.0
    wordcloud==1.9.3
    plotly==5.23.0
    
    # App
    streamlit==1.37.1
    
    # Optional deep learning (comment out if not needed)
    # torch==2.4.0
    # transformers==4.43.4
    """).strip()
open(os.path.join(BASE, "requirements.txt"), "w", encoding="utf-8").write(requirements)

# ---------- .gitignore ----------
gitignore = textwrap.dedent("""\
    # Byte-compiled / cache
    __pycache__/
    *.py[cod]
    *$py.class
    
    # Virtual env
    .venv/
    venv/
    
    # Data & models
    data/*
    !data/.gitkeep
    models/*
    !models/.gitkeep
    reports/*
    !reports/.gitkeep
    
    # Notebook checkpoints
    .ipynb_checkpoints/
    
    # OS files
    .DS_Store
    Thumbs.db
    """).strip()
open(os.path.join(BASE, ".gitignore"), "w", encoding="utf-8").write(gitignore)
open(os.path.join(BASE, "data/.gitkeep"), "w").write("")
open(os.path.join(BASE, "models/.gitkeep"), "w").write("")
open(os.path.join(BASE, "reports/.gitkeep"), "w").write("")

# ---------- README.md ----------
readme = textwrap.dedent("""\
    # AI Echo: Your Smartest Conversational Partner — Sentiment Analysis
    
    End-to-end project for analyzing ChatGPT-style user reviews and classifying sentiment
    (Positive / Neutral / Negative). Includes preprocessing, EDA, classical ML models,
    evaluation, and a Streamlit dashboard for insights + live prediction.
    
    ## 🧱 Project Structure
    ```text
    ai-echo-sentiment/
    ├─ app/
    │  └─ streamlit_app.py           # Interactive dashboard + predictor
    ├─ src/
    │  ├─ data_preprocessing.py      # Cleaning, labeling, train/val split
    │  ├─ train_models.py            # Train/evaluate multiple models, save best
    │  └─ utils.py                   # Helper functions (plots, metrics)
    ├─ data/
    │  └─ chatgpt_style_reviews_dataset.xlsx (place here)  # not tracked
    ├─ models/                       # Saved pipelines (.pkl)
    ├─ reports/                      # EDA images/figures
    ├─ notebooks/                    # Optional: EDA notebooks
    ├─ requirements.txt
    └─ README.md
    ```
    
    ## 📊 Dataset
    Expected file: `data/chatgpt_style_reviews_dataset.xlsx`
    
    **Columns**
    - `date`, `title`, `review`, `rating` (1–5), `username`, `helpful_votes`, `review_length`,
      `platform` (Web/Mobile), `language`, `location`, `version`, `verified_purchase` (Yes/No)
    
    ## 🧪 Quickstart
    1) Create & activate a virtual environment, then install dependencies:
    ```bash
    python -m venv .venv
    .venv\\Scripts\\activate  # on Windows
    # source .venv/bin/activate  # on macOS/Linux
    pip install -r requirements.txt
    ```
    2) Put your dataset at `data/chatgpt_style_reviews_dataset.xlsx`.
    
    3) (One time) Download NLTK assets:
    ```bash
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
    ```
    
    4) Preprocess + split + cache clean data (optional step is integrated into training too):
    ```bash
    python -m src.data_preprocessing --infile data/chatgpt_style_reviews_dataset.xlsx --outfile data/clean_reviews.parquet
    ```
    
    5) Train models and save best pipeline to `models/best_sentiment_model.pkl`:
    ```bash
    python -m src.train_models --data data/clean_reviews.parquet --model_out models/best_sentiment_model.pkl
    ```
    
    6) Launch the Streamlit dashboard:
    ```bash
    streamlit run app/streamlit_app.py
    ```
    
    ## 🧠 Modeling
    - Feature extraction: TF-IDF (word-level n-grams)
    - Models compared: Logistic Regression, Linear SVM, Multinomial Naive Bayes, RandomForest
    - Multiclass metrics: Accuracy, Precision/Recall/F1 (macro), Confusion Matrix, ROC-AUC (OvR)
    
    ## 🔎 Key Dashboard Insights
    - Rating distribution, helpful votes, word clouds (pos vs neg), trends over time
    - Ratings by location, platform, verification status
    - Review length by rating, 1-star keywords, best version by average rating
    - Live sentiment prediction for any text (Positive / Neutral / Negative)
    
    ## 🗂️ Notes
    - By default, sentiment labels are derived from rating:
      - 1–2 → Negative, 3 → Neutral, 4–5 → Positive
    - You can swap in a human-labeled `sentiment` column; the pipeline will auto-detect and use it.
    
    ## 🚀 Deployment
    - Local: `streamlit run app/streamlit_app.py`
    - Cloud: Streamlit Community Cloud or any VM with Python 3.10+
    
    ## 📝 License
    MIT
    """).strip()
open(os.path.join(BASE, "README.md"), "w", encoding="utf-8").write(readme)
