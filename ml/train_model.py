
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.preprocessing import clean_text


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dataset and model paths
DATA_PATH = ROOT_DIR / "data" / "dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "ai_detector.pkl"


def train_model():

    print("Loading dataset...")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Check whether required columns exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    # Remove rows with missing text or labels
    df = df.dropna(subset=["text", "label"])

    # Convert text into string format
    df["text"] = df["text"].astype(str)

    # Clean text using preprocessing function
    df["text"] = df["text"].apply(clean_text)

    # Remove empty text
    df = df[df["text"].str.strip() != ""]

    # Convert text labels into numeric labels
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

    # Map labels to numeric values
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .map(label_mapping)
    )

    # Remove labels that could not be mapped
    df = df.dropna(subset=["label"])

    # Keep only valid binary labels
    df = df[df["label"].isin([0, 1])]

    # Convert labels into integers
    df["label"] = df["label"].astype(int)

    # Separate input text and target labels
    X = df["text"]
    y = df["label"]

    print("Total samples:", len(df))

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create machine learning pipeline
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

    # Create models directory if it does not exist
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the trained model
    joblib.dump(model, MODEL_PATH)

    print("Model trained successfully!")
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_model()
