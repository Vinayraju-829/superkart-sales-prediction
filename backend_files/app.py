
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("superkart_model.joblib")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "SuperKart Sales Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame(data if isinstance(data, list) else [data])
        predictions = model.predict(input_data)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
