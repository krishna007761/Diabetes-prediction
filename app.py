import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('diabetes.csv')

df = load_data()

# Title and Description
st.title("Diabetes Prediction App")
st.write("This app uses logistic regression to predict diabetes based on health data.")

# Display dataset information
st.subheader("Dataset Overview")
st.write(df.head())
st.write(df.describe())
st.write("Missing Values:\n", df.isnull().sum())

# Count of Outcome variable
st.subheader("Outcome Count")
st.write(df['Outcome'].value_counts())


# Split the data
X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy and Confusion Matrix
st.subheader("Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot(plt)

# Classification report
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))
