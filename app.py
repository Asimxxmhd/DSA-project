
import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Load the trained model
model = pickle.load(open('election_data.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        category = request.form['category']
        education = request.form['education']

        # Perform prediction based on selected category and education
        prediction_result = predict_winner(category, education)

        # Pass the prediction result to the result page
        return redirect(url_for('result', result=prediction_result))

    return render_template('index.html')

@app.route('/result/<result>')
def result(result):
    return render_template('result.html', result=result)

def predict_winner(category, education):
    # Create a DataFrame with input data
    input_data = pd.DataFrame({
        'CATEGORY': [category],
        'EDUCATION': [education]
    })

    # Encode categorical columns
    input_data['CATEGORY'] = le.transform(input_data['CATEGORY'])
    input_data['EDUCATION'] = le.transform(input_data['EDUCATION'])

    # Standardize numeric columns
    numeric_columns = ['CATEGORY', 'EDUCATION']
    for col in numeric_columns:
        input_data[col] = std.transform(input_data[col].values.reshape(-1, 1))

    # Use the trained model to make predictions
    prediction = model.predict(input_data)[0]

    return prediction


if __name__ == '__main__':
    app.run(debug=True)
