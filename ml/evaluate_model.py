
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


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dataset and trained model paths
DATA_PATH = ROOT_DIR / "data" / "dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "ai_detector.pkl"


def evaluate_model():

    print("Loading dataset...")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Remove rows with missing text or labels
    df = df.dropna(subset=["text", "label"])

    # Convert text into string format
    df["text"] = df["text"].astype(str)

    # Clean the text using preprocessing function
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

    # Map labels to numeric values
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .map(label_mapping)
    )

    # Remove labels that could not be mapped
    df = df.dropna(subset=["label"])

    # Convert labels into integers
    df["label"] = df["label"].astype(int)

    # Keep only valid labels: 0 = Human, 1 = AI
    df = df[df["label"].isin([0, 1])]

    print("Total samples:", len(df))

    # Separate input text and target labels
    X = df["text"]
    y = df["label"]

    # Use the same train-test split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Loading trained model...")

    # Load the trained machine learning model
    model = joblib.load(MODEL_PATH)

    # Generate predictions on test data
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    print("\nAccuracy:",
          accuracy_score(y_test, predictions))

    print("Precision:",
          precision_score(y_test, predictions))

    print("Recall:",
          recall_score(y_test, predictions))

    print("F1 Score:",
          f1_score(y_test, predictions))

    # Display detailed classification report
    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        target_names=["Human", "AI"]
    ))

    # Display confusion matrix
    print("\nConfusion Matrix:")

    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    evaluate_model()
