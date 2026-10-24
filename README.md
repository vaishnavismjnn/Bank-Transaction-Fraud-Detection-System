# Bank-Transaction-Fraud-Detection-System

Overview

This project uses machine learning to identify potentially fraudulent banking transactions. It analyzes transaction data, compares multiple classification models, and predicts whether a transaction is fraudulent or legitimate.

Features

Processes and analyzes large-scale banking transaction data.

Performs data cleaning, preprocessing, and feature engineering.

Trains and evaluates Logistic Regression, Random Forest, and XGBoost models.

Evaluates performance using ROC-AUC, PR-AUC, precision, recall, F1-score, and a confusion matrix.

Uses classification threshold optimization to balance fraud detection and false positives.

Provides an interactive web application using Streamlit for fraud prediction.

Technologies Used

Programming: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost

Visualization: Matplotlib, Seaborn

Web Application: Streamlit

Environment: Jupyter Notebook

Dataset

The project uses a banking transaction dataset containing transaction details, account balance information, transaction types, and fraud labels.

The target variable is isFraud:

0 – Legitimate transaction

1 – Fraudulent transaction

Machine Learning Models

The following models are trained and compared:

Logistic Regression

Random Forest

XGBoost

XGBoost achieved a ROC-AUC of approximately 99.81% and PR-AUC of approximately 86.22% in the reported evaluation.

Project Workflow

Load and explore the transaction dataset.

Clean and preprocess the data.

Engineer relevant transaction features.

Split the data into training and testing sets.

Train and evaluate machine learning models.

Optimize the classification threshold.

Deploy fraud prediction through a Streamlit application.
