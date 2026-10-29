
from pathlib import Path
import json
import urllib.request
import zipfile

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms+spam+collection.zip"
)

ZIP_PATH = DATA_DIR / "sms_spam_collection.zip"
DATA_PATH = DATA_DIR / "SMSSpamCollection"

MODEL_PATH = MODEL_DIR / "scamshield_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


# ==================================================
# 2. DOWNLOAD DATASET
# ==================================================

def download_dataset():
    """Download and extract the public SMS spam dataset."""

    if DATA_PATH.exists():
        print("Dataset already exists.")
        return

    print("Downloading SMS Spam Collection dataset...")

    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(DATA_DIR)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PATH}"
        )

    print("Dataset downloaded successfully.")


# ==================================================
# 3. LOAD AND CLEAN DATASET
# ==================================================

def load_dataset():
    """Load and prepare labeled SMS messages."""

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    df = df.dropna(subset=["label", "message"]).copy()

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[
        df["label"].isin(["ham", "spam"])
        & df["message"].ne("")
    ].copy()

    # Remove duplicate messages before splitting.
    df = df.drop_duplicates(subset=["message"])

    df["label"] = df["label"].map({
        "ham": "legitimate",
        "spam": "spam",
    })

    if df["label"].nunique() != 2:
        raise ValueError(
            "Expected both legitimate and spam classes."
        )

    print("\nDataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["label"].value_counts())

    return df


# ==================================================
# 4. TRAIN AND EVALUATE MODEL
# ==================================================

def train_model(df):
    """Train the NLP classifier and save evaluation metrics."""

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ----------------------------------------------
    # CALCULATE METRICS
    # ----------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    spam_precision = precision_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0,
    )

    spam_recall = recall_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0,
    )

    spam_f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["legitimate", "spam"],
    )

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Spam precision:  {spam_precision:.4f}")
    print(f"Spam recall:     {spam_recall:.4f}")
    print(f"Spam F1-score:   {spam_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion matrix (legitimate, spam):")
    print(matrix)

    # ----------------------------------------------
    # SAVE METRICS FOR STREAMLIT DASHBOARD
    # ----------------------------------------------

    metrics = {
        "accuracy": float(accuracy),
        "spam_precision": float(spam_precision),
        "spam_recall": float(spam_recall),
        "spam_f1": float(spam_f1),
        "confusion_matrix": matrix.tolist(),
        "test_samples": int(len(y_test)),
        "label_order": ["legitimate", "spam"],
        "dataset": "UCI SMS Spam Collection",
        "test_size": 0.20,
        "random_state": 42,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nMetrics saved to: {METRICS_PATH}")

    # ----------------------------------------------
    # SAVE TRAINED MODEL
    # ----------------------------------------------

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print("\nTraining and evaluation completed successfully.")

    return model


# ==================================================
# 5. MAIN ENTRY POINT
# ==================================================

def main():
    download_dataset()
    df = load_dataset()
    train_model(df)


if __name__ == "__main__":
    main()
