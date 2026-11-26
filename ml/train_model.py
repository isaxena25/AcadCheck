import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.preprocessing import clean_text


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "ai_detector.pkl"


def train_model():

    print("Loading dataset...")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Check required columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    # Remove missing values
    df = df.dropna(subset=["text", "label"])

    # Convert text to string
    df["text"] = df["text"].astype(str)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Remove empty text
    df = df[df["text"].str.strip() != ""]

    # Convert labels to numeric
    label_mapping = {
    "human": 0,
    "Human": 0,
    "human-written": 0,

    "machine": 1,
    "Machine": 1,
    "ai": 1,
    "AI": 1,
    "ai-generated": 1
}

    df["label"] = pd.to_numeric(
        df["label"], errors="coerce"
    )

    df = df.dropna(subset=["label"])

    # Keep only binary labels
    df = df[df["label"].isin([0, 1])]

    X = df["text"]
    y = df["label"].astype(int)

    print("Total samples:", len(df))

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create ML pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    print("Training model...")

    # Train the model
    model.fit(X_train, y_train)

    # Save trained model
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print("Model trained successfully!")
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_model()
