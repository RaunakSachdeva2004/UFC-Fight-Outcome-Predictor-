from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from flask_cors import CORS
import os
from scipy.sparse import issparse

# --- Define the Neural Network Architecture (must match the training script) ---
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu2(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output_layer(x))
        return x

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load Model and Preprocessor ---
model = None
preprocessor = None
MODEL_LOAD_ERROR = False
model_path = 'ufc_prediction_model.pth'
preprocessor_path = 'preprocessor.joblib'

print("--- Server Starting Up ---")
try:
    preprocessor = joblib.load(preprocessor_path)
    print("[SUCCESS] Preprocessor loaded.")

    # A more robust way to determine the model's input feature size
    num_categorical_features = len(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out())
    num_numerical_features = len(preprocessor.named_transformers_['num']['scaler'].get_feature_names_out())
    input_features = num_numerical_features + num_categorical_features
    print(f"Model input feature size calculated: {input_features}")

    model = NeuralNet(input_size=input_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("[SUCCESS] PyTorch model loaded and set to evaluation mode.")

except Exception as e:
    print(f"\n[ERROR] An error occurred while loading files: {e}")
    MODEL_LOAD_ERROR = True

# --- API Routes ---
@app.route('/')
def home():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if MODEL_LOAD_ERROR:
        return jsonify({'error': 'Model or preprocessor not loaded. Please train the model first.'}), 500

    try:
        data = request.json
        input_df = pd.DataFrame([data])
        processed_input = preprocessor.transform(input_df)

        # FIX: The preprocessor might return a sparse matrix. If so, convert to a dense array.
        if issparse(processed_input):
            processed_input = processed_input.toarray()

        input_tensor = torch.tensor(processed_input, dtype=torch.float32)

        with torch.no_grad():
            prediction_proba = model(input_tensor).item()
        
        winner = 'Red Fighter' if prediction_proba > 0.5 else 'Blue Fighter'
        confidence = prediction_proba * 100 if winner == 'Red Fighter' else (1 - prediction_proba) * 100
            
        response = {
            'predicted_winner': winner,
            'confidence_percentage': f'{confidence:.2f}%',
        }
        return jsonify(response)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

