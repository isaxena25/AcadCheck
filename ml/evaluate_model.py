
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from ml.preprocessing import clean_text


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "ai_detector.pkl"


def evaluate_model():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Remove rows with missing text or labels
    df = df.dropna(subset=["text", "label"])

    # Clean the text
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].apply(clean_text)

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

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .map(label_mapping)
    )

    # Remove labels that could not be mapped
    df = df.dropna(subset=["label"])

    # Convert labels to integers
    df["label"] = df["label"].astype(int)

    # Keep only valid labels
    df = df[df["label"].isin([0, 1])]

    print("Total samples:", len(df))

    X = df["text"]
    y = df["label"]

    # Use the same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Loading trained model...")

    model = joblib.load(MODEL_PATH)

    # Generate predictions
    predictions = model.predict(X_test)

    print("\nAccuracy:",
          accuracy_score(y_test, predictions))

    print("Precision:",
          precision_score(y_test, predictions))

    print("Recall:",
          recall_score(y_test, predictions))

    print("F1 Score:",
          f1_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Human", "AI"]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    evaluate_model()
