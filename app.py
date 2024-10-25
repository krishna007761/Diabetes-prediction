import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('diabetes.csv')

df = load_data()

# Display the dataset
st.title("Diabetes Prediction App")
st.write("This app predicts if a person is diabetic based on medical data.")
st.write("### Dataset Overview")
st.write(df.head())
st.write("### Dataset Statistics")
st.write(df.describe())

# Data visualization
st.write("### Correlation Matrix")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)

st.write("### Glucose Level Distribution for Diabetic vs Non-Diabetic")
plt.figure(figsize=(8, 6))
sns.histplot(df[df['Outcome'] == 1]['Glucose'], color='red', label='Diabetic', kde=True)
sns.histplot(df[df['Outcome'] == 0]['Glucose'], color='green', label='Non-Diabetic', kde=True)
plt.legend()
plt.title('Glucose Level Distribution')
st.pyplot(plt)

# Split data into features and target
X = df.drop(columns='Outcome')
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and evaluation metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.write("### Model Performance")
st.write(f"Accuracy: {accuracy * 100:.2f}%")
st.write("#### Confusion Matrix")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot(plt)

st.write("#### Classification Report")
report = classification_report(y_test, y_pred)
st.text_area("Classification Report", report, height=200)

# Prediction form
st.write("### Make a Prediction")
def get_user_input():
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.slider("Glucose", min_value=0, max_value=200, value=120)
    blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=122, value=70)
    skin_thickness = st.slider("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.slider("Insulin", min_value=0, max_value=846, value=79)
    bmi = st.slider("BMI", min_value=0.0, max_value=67.1, value=32.0)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.slider("Age", min_value=21, max_value=100, value=30)
    return np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

user_input = get_user_input()
user_input_scaled = scaler.transform(user_input)

if st.button("Predict"):
    prediction = model.predict(user_input_scaled)
    if prediction[0] == 1:
        st.write("The model predicts that this person is **Diabetic**.")
    else:
        st.write("The model predicts that this person is **Not Diabetic**.")
