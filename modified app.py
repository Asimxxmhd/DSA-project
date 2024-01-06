import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Flask application
app = Flask(__name__)

# Load the trained model and label encoders
with open('election_data_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('le_category.pkl', 'rb') as le_category_file:
    le_category = pickle.load(le_category_file)

with open('le_education.pkl', 'rb') as le_education_file:
    le_education = pickle.load(le_education_file)

@app.route('/')
def home():
    return render_template('index.html')

# Add route for prediction result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        category = request.form['category']
        education = request.form['education']
        criminal_cases = float(request.form['criminal_cases'])
        age = float(request.form['age'])

        # Encode categorical variables
        category_encoded = le_category.transform([category])[0]
        education_encoded = le_education.transform([education])[0]

        # Prepare the input data for prediction
        input_data = {
            'CATEGORY': [category_encoded],
            'EDUCATION': [education_encoded],
            'CRIMINAL CASES': [criminal_cases],
            'AGE': [age]
            # Add other columns as needed based on your model features
        }
       # Make the prediction using the trained model
        prediction = model.predict(input_data)[0]

        # Convert the prediction to a human-readable format if needed
        prediction_result = "Winner" if prediction == 1 else "Loser"

        # Render the result page with the prediction
        return render_template('result.html', prediction_result=prediction_result)



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
