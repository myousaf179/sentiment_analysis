from flask import Flask, request, jsonify, send_from_directory
import joblib
import os

# Flask app init
app = Flask(__name__, static_folder='.', static_url_path='')

# Load trained pipeline (trainable_file.joblib)
model_path = os.path.join(os.path.dirname(__file__), "trainable_file.joblib")
pipeline = joblib.load(model_path)

# Serve HTML page
@app.route("/")
def index():
    return send_from_directory(".", "MovieReviewApp.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        review_text = data.get("review", "")

        if not review_text.strip():
            return jsonify({"error": "Empty review text"}), 400

        # Predict (1 = Positive, 0 = Negative)
        pred_label = pipeline.predict([review_text])[0]

        return jsonify({
            "prediction": "Positive" if pred_label == 1 else "Negative"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
