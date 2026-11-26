import joblib

from pathlib import Path

from ml.preprocessing import clean_text


ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "ai_detector.pkl"


def predict_text(text):

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Clean input text
    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("Please provide valid text.")

    # Predict class
    prediction = model.predict([cleaned_text])[0]

    # Get probability
    probabilities = model.predict_proba(
        [cleaned_text]
    )[0]

    ai_probability = probabilities[1] * 100

    if prediction == 1:
        result = "AI-generated"
    else:
        result = "Human-written"

    return {
        "prediction": result,
        "ai_likelihood": round(
            float(ai_probability), 2
        )
    }


if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is transforming
    modern education through technology.
    """

    result = predict_text(sample_text)

    print(result)
