# ML Assignment 2 – Heart Disease Classification

---

## A. Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early and accurate prediction of heart disease can significantly improve patient outcomes and assist healthcare professionals in clinical decision-making.

The objective of this project is to implement and compare multiple machine learning classification models to predict whether a patient has heart disease based on clinical and diagnostic features. The project also demonstrates end-to-end machine learning deployment using a Streamlit web application.

---

## B. Dataset Description

- **Dataset Name:** Heart Failure Prediction Dataset  
- **Source:** Kaggle  
- **Type:** Binary Classification  
- **Total Instances:** 918  
- **Number of Features:** 13  
- **Target Variable:** `HeartDisease`  
  - 0 → No Heart Disease  
  - 1 → Heart Disease Present  

### Preprocessing Performed

- One-hot encoding for categorical features  
- Feature scaling using StandardScaler  
- Stratified train-test split (80% training, 20% testing)  
- No missing values present in the dataset  

The dataset satisfies the assignment requirement of minimum 12 features and minimum 500 instances.

---

## C. Models Used and Evaluation Metrics

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.8859 | 0.9297 | 0.8716 | 0.9314 | 0.9005 | 0.7694 |
| Decision Tree | 0.7989 | 0.7923 | 0.7982 | 0.8529 | 0.8246 | 0.5914 |
| KNN | 0.8859 | 0.9360 | 0.8857 | 0.9118 | 0.8986 | 0.7686 |
| Naive Bayes | 0.9130 | 0.9451 | 0.9300 | 0.9118 | 0.9208 | 0.8246 |
| Random Forest (Ensemble) | 0.8696 | 0.9304 | 0.8750 | 0.8922 | 0.8835 | 0.7356 |
| XGBoost (Ensemble) | 0.8587 | 0.9219 | 0.8725 | 0.8725 | 0.8725 | 0.7140 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieved strong recall (93.14%), indicating effective identification of heart disease cases. Provides stable and interpretable baseline performance. |
| Decision Tree | Recorded the lowest accuracy and MCC. Prone to overfitting and weaker generalization compared to ensemble methods. |
| KNN | Delivered competitive performance with high AUC (0.9360). Performance depends heavily on feature scaling and choice of K. |
| Naive Bayes | Achieved the best overall results with highest accuracy (91.30%), highest AUC (0.9451), and highest MCC (0.8246). Demonstrated strong class separability. |
| Random Forest (Ensemble) | Provided balanced performance with reduced variance compared to a single decision tree. Good overall generalization. |
| XGBoost (Ensemble) | Showed strong predictive capability with competitive AUC (0.9219). Performs well but slightly lower accuracy compared to Naive Bayes on this dataset. |

---

## Streamlit Application Features

The deployed Streamlit application includes:

- CSV dataset upload option  
- Model selection dropdown  
- Run Model button for controlled execution  
- Display of evaluation metrics  
- Detailed classification report  
- Confusion matrix visualization  
- Theme-adaptive interface (Light/Dark mode support)  

---

## Project Structure

ML_Assignment_2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│ ├── Logistic_Regression.pkl
│ ├── Decision_Tree.pkl
│ ├── KNN.pkl
│ ├── Naive_Bayes.pkl
│ ├── Random_Forest.pkl
│ ├── XGBoost.pkl
│ ├── scaler.pkl
│ └── feature_columns.pkl


---

## Deployment

The application has been deployed using Streamlit Community Cloud.

Steps:
1. Push repository to GitHub
2. Connect repository to Streamlit Cloud
3. Select `app.py`
4. Deploy the application

---

## Conclusion

This project successfully demonstrates the implementation and comparison of six machine learning classification models for heart disease prediction.

Among all models, **Naive Bayes achieved the best overall performance** based on Accuracy, AUC, and MCC metrics for this dataset.

The project fulfills all assignment requirements including model implementation, evaluation, Streamlit application development, and deployment.
