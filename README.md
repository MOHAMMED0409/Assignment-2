# ML Assignment 2 – Heart Disease Classification

## Machine Learning Classification & Streamlit Deployment

---

## 1. Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early detection and accurate prediction can significantly improve patient outcomes and reduce health risks.

The objective of this project is to implement and compare multiple machine learning classification models to predict whether a patient has heart disease based on clinical attributes. The project also demonstrates an end-to-end machine learning workflow including preprocessing, model evaluation, web application development using Streamlit, and deployment on Streamlit Community Cloud.

---

## 2. Dataset Description

* **Dataset Name:** Heart Failure Prediction Dataset
* **Source:** Kaggle
* **Type:** Binary Classification
* **Total Instances:** 918
* **Number of Features:** 13
* **Target Variable:** `HeartDisease`

  * 0 → No Heart Disease
  * 1 → Heart Disease Present

### Preprocessing Steps

* One-hot encoding for categorical variables
* Feature scaling using StandardScaler
* Stratified train-test split (80% training, 20% testing)
* No missing values present

The dataset satisfies the assignment requirement of minimum 12 features and minimum 500 instances.

---

## 3. Models Implemented

The following six classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

---

## 4. Evaluation Metrics

Each model was evaluated using:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## 5. Model Comparison Table

| ML Model            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.8859   | 0.9297 | 0.8716    | 0.9314 | 0.9005   | 0.7694 |
| Decision Tree       | 0.7989   | 0.7923 | 0.7982    | 0.8529 | 0.8246   | 0.5914 |
| K-Nearest Neighbors | 0.8859   | 0.9360 | 0.8857    | 0.9118 | 0.8986   | 0.7686 |
| Naive Bayes         | 0.9130   | 0.9451 | 0.9300    | 0.9118 | 0.9208   | 0.8246 |
| Random Forest       | 0.8696   | 0.9304 | 0.8750    | 0.8922 | 0.8835   | 0.7356 |
| XGBoost             | 0.8587   | 0.9219 | 0.8725    | 0.8725 | 0.8725   | 0.7140 |

---

## 6. Observations on Model Performance

| ML Model            | Observation                                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Strong recall (93.14%) and balanced performance. Reliable and interpretable baseline model.               |
| Decision Tree       | Lowest performance among all models. Prone to overfitting and weaker generalization.                      |
| KNN                 | Competitive performance with high AUC. Sensitive to feature scaling and choice of K.                      |
| Naive Bayes         | Best overall performance with highest accuracy (91.30%), highest AUC (0.9451), and highest MCC (0.8246).  |
| Random Forest       | Stable ensemble model with good generalization and balanced precision-recall tradeoff.                    |
| XGBoost             | Strong boosting-based model with competitive performance though slightly lower accuracy than Naive Bayes. |

---

## 7. Streamlit Application Features

The deployed Streamlit application includes:

* CSV dataset upload
* Model selection dropdown
* Run Model button
* Display of evaluation metrics
* Detailed classification report
* Confusion matrix visualization
* Theme-adaptive interface (Light/Dark mode support)

---

## 8. Project Structure

```
ML_Assignment_2/
│
├── app.py
├── ML_Assignment_2.ipynb
├── requirements.txt
├── README.md
│
├── data/
│   └── heart.csv
│
└── model/
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── KNN.pkl
    ├── Naive_Bayes.pkl
    ├── Random_Forest.pkl
    ├── XGBoost.pkl
    ├── scaler.pkl
    └── feature_columns.pkl
```

---

## 9. Installation

### Step 1: Clone Repository

```
git clone <your-github-repository-link>
cd ML_Assignment_2
```

### Step 2: Create Virtual Environment (Recommended)

```
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

---

## 10. Running the Application

```
streamlit run app.py
```

The application will open at:

```
http://localhost:8501
```

---

## 11. Deployment

1. Push the repository to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Select "New App"
4. Choose repository and branch
5. Select `app.py`
6. Click Deploy

---

## 12. Conclusion

This project successfully demonstrates the implementation and comparison of six machine learning classification models for heart disease prediction.

Based on the evaluation metrics, **Naive Bayes achieved the best overall performance** on this dataset.

The project fulfills all assignment requirements including model implementation, evaluation, Streamlit application development, and deployment.
