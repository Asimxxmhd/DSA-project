from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = 'election_model.pkl'
with open(model_filename, 'rb') as model_file:
    clf = pickle.load(model_file)

# Load the label encoder
encoder_filename = 'label_encoder.pkl'
with open(encoder_filename, 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the scaler
scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        category = request.form['category']
        age = float(request.form['age'])
        criminal_cases = float(request.form['criminal_cases'])
        education = request.form['education']

        # Transform user input using label encoder and scaler
        category_encoded = label_encoder.transform([category])[0]
        education_encoded = label_encoder.transform([education])[0]

        # Create a numpy array with the input values
        input_data = np.array([[category_encoded, age, criminal_cases, education_encoded]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction using the trained model
        prediction = clf.predict(input_data_scaled)[0]

        # Determine the prediction result (Win or Lose)
        result = "Win" if prediction == 1 else "Lose"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
