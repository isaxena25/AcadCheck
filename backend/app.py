
from flask import Flask, request, jsonify
from flask_cors import CORS

from ml.predict import predict_text
from backend.document_parser import extract_text


app = Flask(__name__)

CORS(app)

# Maximum uploaded file size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/")
def home():

    return jsonify({
        "message": "Welcome to AcadCheck API",
        "status": "running"
    })


# API for checking plain text
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({
                "error": "Text is required"
            }), 400

        text = data["text"]

        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "error": "Please provide valid text"
            }), 400

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


# API for uploading academic documents
@app.route("/upload", methods=["POST"])
def upload_document():

    try:

        # Check whether a file was uploaded
        if "file" not in request.files:
            return jsonify({
                "error": "Please upload a document"
            }), 400

        file = request.files["file"]

        # Check whether filename is empty
        if file.filename == "":
            return jsonify({
                "error": "No file selected"
            }), 400

        filename = file.filename

        # Read uploaded file bytes
        file_bytes = file.read()

        # Extract text from PDF, DOCX or TXT
        extracted_text = extract_text(
            file_bytes,
            filename
        )

        # Check whether document contains text
        if not extracted_text.strip():
            return jsonify({
                "error": "No readable text found in document"
            }), 400

        # Send extracted text to AI detector
        result = predict_text(extracted_text)

        # Return document analysis result
        return jsonify({
            "filename": filename,
            "characters_analyzed": len(extracted_text),
            "ai_detection": result
        })

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

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

