import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- THEME ADAPTIVE CSS ----------------
st.markdown("""
<style>

/* Titles */
.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    opacity: 0.7;
    margin-bottom: 25px;
}

/* Section Headers */
.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 20px;
}

/* Metric Cards */
.metric-card {
    padding: 28px;
    border-radius: 12px;
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.25);
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    transition: all 0.2s ease-in-out;
    text-align: center;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

.metric-title {
    font-size: 14px;
    font-weight: 500;
    opacity: 0.7;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
}

/* Adaptive Button */
.stButton > button {
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

/* DataFrame Styling */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(128,128,128,0.3);
    border-radius: 10px;
    overflow: hidden;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 50px;
    font-size: 13px;
    opacity: 0.6;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">Heart Disease Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Classification Models – Assignment 2</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

model_name = st.sidebar.selectbox(
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

run_button = st.sidebar.button("Run Model")

# ---------------- MAIN EXECUTION ----------------
if run_button:

    if uploaded_file is None:
        st.warning("Please upload a dataset from the sidebar.")
    else:
        try:
            with st.spinner("Running model..."):

                data = pd.read_csv(uploaded_file)

                # Load model artifacts
                model = joblib.load(f"model/{model_name}.pkl")
                scaler = joblib.load("model/scaler.pkl")
                feature_columns = joblib.load("model/feature_columns.pkl")

                # Separate target
                if "HeartDisease" in data.columns:
                    y_true = data["HeartDisease"]
                    data = data.drop("HeartDisease", axis=1)
                else:
                    y_true = None

                # Preprocess
                data_processed = pd.get_dummies(data, drop_first=True)
                data_processed = data_processed.reindex(columns=feature_columns, fill_value=0)
                data_scaled = scaler.transform(data_processed)

                predictions = model.predict(data_scaled)

            st.success("Model executed successfully.")

            if y_true is not None:

                accuracy = accuracy_score(y_true, predictions)
                report_dict = classification_report(y_true, predictions, output_dict=True)

                # ---------------- PERFORMANCE OVERVIEW ----------------
                st.markdown('<div class="section-title">Model Performance Overview</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Accuracy</div>
                        <div class="metric-value">{accuracy:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Precision</div>
                        <div class="metric-value">{report_dict['weighted avg']['precision']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Recall</div>
                        <div class="metric-value">{report_dict['weighted avg']['recall']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # ---------------- DETAILED REPORT ----------------
                st.markdown('<div class="section-title">Detailed Classification Report</div>', unsafe_allow_html=True)

                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format("{:.3f}"))

                st.markdown("---")

                # ---------------- CONFUSION MATRIX ----------------
                st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

                cm = confusion_matrix(y_true, predictions)

                fig, ax = plt.subplots(figsize=(6,5))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"],
                    ax=ax
                )

                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

                st.pyplot(fig)

            else:
                st.info("Predictions generated successfully (no target column found).")

        except Exception as e:
            st.error(f"Error: {e}")