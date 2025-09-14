from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Load the trained model and preprocessor ---
try:
    model = tf.keras.models.load_model('ufc_prediction_model.h5')
    preprocessor = joblib.load('preprocessor.joblib')
    print("Model and preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    model = None
    preprocessor = None

@app.route('/')
def home():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    Receives fighter stats as JSON, preprocesses them,
    and returns the prediction.
    """
    if not model or not preprocessor:
        return jsonify({'error': 'Model or preprocessor not loaded. Please train the model first.'}), 500

    try:
        # Get data from POST request
        data = request.json
        print("Received data:", data)

        # Create a pandas DataFrame from the incoming JSON
        # The structure must match the columns used during training
        input_df = pd.DataFrame([data])
        
        # Ensure column order is the same as in training
        # This is crucial for the preprocessor to work correctly
        training_columns = [
            'R_Height_cms', 'R_Weight_lbs', 'R_Reach_cms', 'R_wins', 'R_losses',
            'B_Height_cms', 'B_Weight_lbs', 'B_Reach_cms', 'B_wins', 'B_losses',
            'R_Stance', 'B_Stance'
        ]
        input_df = input_df[training_columns]

        # Preprocess the input data using the loaded preprocessor
        processed_input = preprocessor.transform(input_df)

        # Make prediction
        prediction_proba = model.predict(processed_input)[0][0]
        
        # Determine winner based on probability
        if prediction_proba > 0.5:
            winner = 'Red Fighter'
            confidence = prediction_proba * 100
        else:
            winner = 'Blue Fighter'
            confidence = (1 - prediction_proba) * 100
            
        # Create response
        response = {
            'predicted_winner': winner,
            'confidence_percentage': f'{confidence:.2f}%',
            'raw_prediction': f'{prediction_proba:.4f}'
        }
        return jsonify(response)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
