from flask import Flask, request, jsonify
from flask_cors import CORS

from ml.predict import predict_text


app = Flask(__name__)

CORS(app)


@app.route("/")
def home():

    return jsonify({
        "message": "Welcome to AcadCheck API",
        "status": "running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({
                "error": "Text is required"
            }), 400

        text = data["text"]

        result = predict_text(text)

        return jsonify(result)

    except FileNotFoundError:

        return jsonify({
            "error": "Model not trained yet"
        }), 500

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )
