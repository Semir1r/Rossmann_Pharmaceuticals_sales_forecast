from flask import Flask, request, jsonify, render_template
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

# Home route with a form
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get input values from the form
            input_data = request.form.getlist("data[]")
            input_data = [float(value) for value in input_data]

            # Validate input size (30 values)
            if len(input_data) != 30:
                return render_template("index.html", error="Please enter exactly 30 values.")

            # Convert to numpy array and scale
            data = np.array(input_data).reshape(-1, 1)
            scaled_data = scaler.transform(data)

            # Reshape for LSTM input
            reshaped_data = scaled_data.reshape(1, 30, 1)

            # Predict sales
            prediction = model.predict(reshaped_data)
            predicted_sales = scaler.inverse_transform(prediction)

            return render_template("index.html", prediction=round(predicted_sales[0][0], 2))
        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {e}")

    # Render the form initially
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
