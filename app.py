from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Expected sensor columns
sensor_columns = ['ENS160 R2 (Ohm)', 'ENS160 R3 (Ohm)', 'SGP40 R (Ohm)', 
                  'Sen0566 R (Ohm)', 'MQ138 R (Ohm)', 'TGS2602 R (Ohm)', 
                  'MiCS5524 R (Ohm)']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    df = pd.read_excel(file)
    
    # Ensure only existing sensor columns are used
    df = df[sensor_columns].apply(pd.to_numeric, errors="coerce")
    
    # Standardize and predict
    X = scaler.transform(df)
    predictions = svm_model.predict_proba(X)[:, 1]
    
    # Compute final likelihood
    final_likelihood = round(np.mean(predictions) * 100, 2)

    return jsonify({"Lung Cancer Likelihood (%)": final_likelihood})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
