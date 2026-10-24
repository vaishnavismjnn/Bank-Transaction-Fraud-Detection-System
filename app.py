
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Bank Transaction Fraud Detection")
st.write(
    "Analyze a bank transaction using your trained machine-learning model."
)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================
MODEL_PATH = Path(__file__).resolve().parent / "best_fraud_model_tuned.pkl"


@st.cache_resource
def load_model_artifact(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(
        "Place best_fraud_model_tuned.pkl in the same folder as app.py."
    )
    st.stop()

try:
    artifact = load_model_artifact(str(MODEL_PATH))

    model = artifact["model"]
    threshold = float(artifact["threshold"])
    features = list(artifact["features"])

except Exception as e:
    st.error("The saved model artifact could not be loaded correctly.")
    st.code(f"{type(e).__name__}: {e}")
    st.info(
        "Check that the PKL file contains the keys: model, threshold, "
        "and features."
    )
    st.stop()

# The scaler must be the same scaler used to transform training data.
scaler = artifact.get("scaler")
type_classes = artifact.get("type_classes")
p99 = artifact.get("p99")

if scaler is None:
    st.error(
        "The saved artifact does not contain the training scaler. "
        "Save the original fitted scaler in the notebook before running "
        "this app."
    )
    st.stop()

if not type_classes:
    st.error(
        "Transaction type classes are missing from the saved artifact. "
        "Save le.classes_.tolist() as type_classes in the notebook."
    )
    st.stop()

if p99 is None:
    st.error(
        "The amount percentile threshold (p99) is missing from the "
        "saved artifact. Save the same p99 value used during training."
    )
    st.stop()

st.success("Trained model loaded successfully.")

# ==========================================
# 3. MODEL INFORMATION
# ==========================================
with st.expander("Model information"):
    st.write("Model:", artifact.get("model_name", type(model).__name__))
    st.write("Decision threshold:", threshold)
    st.write("Number of input features:", len(features))

# ==========================================
# 4. TRANSACTION INPUT FORM
# ==========================================
st.subheader("Enter Transaction Details")

with st.form("fraud_detection_form"):

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction step",
            min_value=0,
            value=100,
            step=1
        )

        transaction_type = st.selectbox(
            "Transaction type",
            options=type_classes
        )

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        oldbalanceOrg = st.number_input(
            "Origin balance before transaction",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

    with col2:
        newbalanceOrig = st.number_input(
            "Origin balance after transaction",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        oldbalanceDest = st.number_input(
            "Destination balance before transaction",
            min_value=0.0,
            value=2000.0,
            step=500.0
        )

        newbalanceDest = st.number_input(
            "Destination balance after transaction",
            min_value=0.0,
            value=7000.0,
            step=500.0
        )

    submitted = st.form_submit_button(
        "Detect Fraud",
        use_container_width=True
    )

# ==========================================
# 5. FEATURE ENGINEERING AND PREDICTION
# ==========================================
if submitted:
    try:
        # Apply the feature engineering used in the notebook.
        hour = int(step) % 24

        transaction_type_encoded = list(type_classes).index(
            transaction_type
        )

        transaction_features = {
            "step": step,
            "amount": amount,
            "log_amount": np.log1p(amount),
            "is_high_amount": int(amount > float(p99)),
            "hour": hour,
            "is_night": int(hour in [0, 1, 2, 3, 4, 5, 22, 23]),
            "balance_diff_orig": oldbalanceOrg - newbalanceOrig,
            "balance_diff_dest": newbalanceDest - oldbalanceDest,
            "type_enc": transaction_type_encoded
        }

        # Preserve the exact feature order expected by the model.
        input_df = pd.DataFrame([transaction_features])
        missing_features = [
            feature for feature in features
            if feature not in input_df.columns
        ]

        if missing_features:
            st.error(
                "The app does not generate all features required by "
                "your trained model."
            )
            st.write("Missing model features:", missing_features)
            st.info(
                "Update the input feature engineering to match the "
                "training notebook exactly."
            )
            st.stop()

        input_df = input_df[features]

        # Apply the fitted scaler from model training.
        scaled_values = scaler.transform(input_df)

        scaled_input = pd.DataFrame(
            scaled_values,
            columns=features
        )

        # Predict fraud probability.
        probability = float(
            model.predict_proba(scaled_input)[0, 1]
        )

        prediction = (
            "Fraud"
            if probability >= threshold
            else "Not Fraud"
        )

        # ======================================
        # 6. DISPLAY RESULTS
        # ======================================
        st.divider()
        st.subheader("Prediction Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Predicted Class",
            prediction
        )

        result_col2.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

        result_col3.metric(
            "Decision Threshold",
            f"{threshold:.2%}"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        if prediction == "Fraud":
            st.error(
                "⚠️ This transaction is flagged as potentially fraudulent."
            )
        else:
            st.success(
                "✅ This transaction is classified as non-fraudulent."
            )

        with st.expander("View model input features"):
            st.dataframe(input_df, use_container_width=True)

        st.caption(
            "This prediction is an estimate, not a guarantee. "
            "Additional verification is needed for real banking decisions."
        )

    except Exception as e:
        st.error("An error occurred while predicting the transaction.")
        st.code(f"{type(e).__name__}: {e}")
