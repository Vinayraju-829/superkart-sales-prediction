
from flask import Flask, request, jsonify
import pandas as pd
import joblib

superkart_api = Flask(__name__)
model = joblib.load("superkart_model.joblib")

@superkart_api.get("/")
def home():
    return jsonify({"message": "SuperKart Sales Prediction API is running"})

@superkart_api.post("/v1/predict")
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)[0]
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@superkart_api.post("/v1/predictbatch")
def predict_batch():
    try:
        data = request.get_json()
        input_data = pd.DataFrame(data)
        predictions = model.predict(input_data)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
