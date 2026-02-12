import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Heart Disease Classification", layout="wide")

st.title("Heart Disease Prediction – ML Assignment 2")

st.markdown("Upload test dataset and select a model to evaluate.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic_Regression",
        "Decision_Tree",
        "KNN",
        "Naive_Bayes",
        "Random_Forest",
        "XGBoost"
    ]
)

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Load saved objects
    model = joblib.load(f"model/{model_name}.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")

    # Preprocess uploaded data
    data = pd.get_dummies(data, drop_first=True)
    data = data.reindex(columns=feature_columns, fill_value=0)
    data_scaled = scaler.transform(data)

    predictions = model.predict(data_scaled)

    st.subheader("Predictions")
    st.write(predictions)

    # If target column exists
    if "HeartDisease" in data.columns:
        y_true = data["HeartDisease"]

        st.subheader("Classification Report")
        st.text(classification_report(y_true, predictions))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
