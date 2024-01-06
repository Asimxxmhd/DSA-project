import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = 'D:/Demo_flask/Project_DSA/election_dataset.csv'
election_data = pd.read_csv(file_path)

# Replace '\n' in column names with spaces
election_data.columns = election_data.columns.str.replace('\n', ' ')

# Replace 'Not available' with 0 in the 'CRIMINAL CASES' column
election_data['CRIMINAL CASES'].replace('Not Available', 0, inplace=True)

# Convert columns to numeric types
election_data['CRIMINAL CASES'] = pd.to_numeric(election_data['CRIMINAL CASES'], errors='coerce')

election_data['EDUCATION'] = election_data['EDUCATION'].replace(['Not Available','Others'],'Illiterate')
election_data['EDUCATION'] = election_data['EDUCATION'].replace(['Post Graduate\n'],'Post Graduate')
election_data['EDUCATION'] = election_data['EDUCATION'].replace(['5th Pass','8th Pass'],'Illiterate')

# Impute missing values in numeric columns with the median

numeric_columns = ['AGE', 'CRIMINAL CASES']
for col in numeric_columns:
    election_data[col].fillna(election_data[col].median(), inplace=True)

# Impute missing values in categorical columns with the most frequent category

categorical_columns = ['GENDER', 'CATEGORY', 'EDUCATION']
for col in categorical_columns:
    election_data[col].fillna(election_data[col].mode()[0], inplace=True)

# Impute missing values in 'SYMBOL' with 'Unknown'
election_data['SYMBOL'].fillna('Unknown', inplace=True)

# Clean and convert 'ASSETS' and 'LIABILITIES' columns
columns_to_fill = ['ASSETS', 'LIABILITIES']

for column in columns_to_fill:
    # Replace non-numeric characters with an empty string using regex
    election_data[column] = election_data[column].replace('[^0-9.]', '', regex=True)

    # Convert the column to numeric type
    election_data[column] = pd.to_numeric(election_data[column], errors='coerce')

    # Impute missing values with the median
    median_value = election_data[column].median()
    election_data[column].fillna(median_value, inplace=True)

# Encode categorical columns
le_gender = LabelEncoder()
le_party = LabelEncoder()
le_category = LabelEncoder()
le_education = LabelEncoder()

election_data['GENDER'] = le_gender.fit_transform(election_data['GENDER'])
election_data['PARTY'] = le_gender.fit_transform(election_data['PARTY'])
election_data['CATEGORY'] = le_category.fit_transform(election_data['CATEGORY'])
election_data['EDUCATION'] = le_education.fit_transform(election_data['EDUCATION'])

# Drop specified columns
columns_to_drop = ['STATE', 'CONSTITUENCY', 'NAME', 'SYMBOL']
election_data.drop(columns_to_drop, axis=1, inplace=True)

# Standardize numeric columns
std_scaler = StandardScaler()

cols_to_standardize = ['PARTY', 'AGE', 'CRIMINAL CASES', 'CATEGORY', 'GENDER', 'TOTAL VOTES', 'TOTAL ELECTORS', 'ASSETS', 'LIABILITIES']
for col in cols_to_standardize:
    election_data[col] = std_scaler.fit_transform(election_data[col].values.reshape(-1, 1))

# Split the cleaned data into training and testing sets, 'WINNER' is the target variable
X = election_data.drop('WINNER', axis=1)
y = election_data['WINNER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)
model = rf_classifier.fit(X_train, y_train)

# Save the trained model and label encoders
with open('election_data_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('le_category.pkl', 'wb') as le_category_file:
    pickle.dump(le_category, le_category_file)

with open('le_education.pkl', 'wb') as le_education_file:
    pickle.dump(le_education, le_education_file)

print("Model and label encoders saved.")