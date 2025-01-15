from flask import Flask, request, jsonify
import numpy as np
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Paths to the model and scaler
MODEL_PATH = "notebooks/lstm_model_store_1.pkl"
SCALER_PATH = "notebooks/scaler_store_1.pkl"

# Load the serialized model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")

# Root endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify("Welcome to the LSTM Sales Prediction API!")

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_sales():
    try:
        # Parse the input JSON request
        input_data = request.json
        if not input_data or "data" not in input_data:
            return jsonify({"error": "Missing 'data' field in the request."}), 400

        # Extract and validate input data
        data = np.array(input_data["data"]).reshape(-1, 1)
        if data.shape[0] != 30:  # Assuming window_size=30
            return jsonify({
                "error": f"Input data must have 30 values (sliding window size). Provided: {data.shape[0]}"
            }), 400

        # Scale input data
        scaled_data = scaler.transform(data)

        # Reshape for LSTM input
        reshaped_data = scaled_data.reshape(1, 30, 1)

        # Make prediction
        prediction = model.predict(reshaped_data)

        # Inverse transform the prediction
        predicted_sales = scaler.inverse_transform(prediction)

        # Return the prediction
        return jsonify({"predicted_sales": float(predicted_sales[0][0])})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
