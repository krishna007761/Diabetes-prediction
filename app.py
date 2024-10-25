from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load and preprocess the dataset
def load_data():
    df = pd.read_csv('diabetes.csv')
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_data = [float(x) for x in request.form.values()]
    X_train, X_test, y_train, y_test = load_data()
    model, scaler = train_model(X_train, y_train)
    X_test_scaled = scaler.transform([input_data])
    prediction = model.predict(X_test_scaled)
    return f'Prediction: {"Diabetic" if prediction[0] == 1 else "Non-Diabetic"}'

if __name__ == '__main__':
    app.run(debug=True)
