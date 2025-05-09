from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load the trained Keras model
model = load_model("best_model(1).h5")

# Example list of expected features (must match training order)
FEATURE_NAMES = [
    "Screenshot Clarity", "Password Protection", "Data Backup", "Encryption Usage",
    "Firewall Enabled", "Patch Management", "Authentication Strength", "Access Control",
    "Malware Detection", "Incident Response", "User Training", "Remote Access Policy",
    "Network Segmentation", "Email Filtering", "Physical Security"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", [])

        if len(features) != len(FEATURE_NAMES):
            return jsonify({
                "error": f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"
            }), 400

        input_array = np.array([features])  # Shape: (1, n_features)
        prediction = model.predict(input_array)[0][0]
        is_secure = prediction >= 0.5  # Binary classification

        return jsonify({
            "prediction": "Secure" if is_secure else "Insecure",
            "score": float(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
