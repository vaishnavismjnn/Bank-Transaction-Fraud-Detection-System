
# Bank Transaction Fraud Detection System

## Overview
A machine learning project that detects potentially fraudulent banking transactions using transaction data and classification algorithms.

## Features
- Analyzed over 6.3 million banking transaction records.
- Performed data cleaning, preprocessing, and feature engineering.
- Trained and compared Logistic Regression, Random Forest, and XGBoost models.
- Evaluated models using ROC-AUC, PR-AUC, precision, recall, and F1-score.
- Optimized classification thresholds to balance fraud detection and false positives.
- Developed an interactive Streamlit application for fraud prediction.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit
- Jupyter Notebook

## Dataset
The dataset contains banking transaction details, transaction types, account balances, and fraud labels.

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

## Model Performance
The XGBoost model achieved the following results in the reported evaluation:

- ROC-AUC: 99.81%
- PR-AUC: 86.22%

Precision and recall vary depending on the classification threshold.

## Project Workflow
1. Data loading and exploratory data analysis
2. Data cleaning and preprocessing
3. Feature engineering
4. Model training and evaluation
5. Threshold optimization
6. Fraud prediction through Streamlit

## Installation

Install the required libraries:

```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn joblib
```

## Run the Application

```bash
streamlit run app.py
```

Ensure the trained model file and required preprocessing artifacts are available before running the application.

## Future Enhancements
- Improve precision while maintaining high fraud recall.
- Perform further hyperparameter tuning.
- Evaluate model performance on unseen transactions.

## Disclaimer
This project is developed for educational purposes and is not intended to replace professional banking fraud detection systems.
