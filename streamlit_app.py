
import streamlit as st
import pandas as pd
import joblib
from lightgbm import LGBMClassifier

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Prediction App")

# 📘 Model Summary
with st.expander("🧠 Model Summary"):
    st.markdown("""
    **Models Used:**
    - `XGBoost`: Gradient boosting model with GridSearch tuning.
    - `LightGBM`: Fast and memory-efficient boosting model.
    - `CatBoost`: Great with categorical data, minimal preprocessing.
    - `VotingClassifier`: Ensemble of the three to enhance performance.

    **Bias Handling:**
    - SMOTEENN used to handle class imbalance (combines oversampling and cleaning).

    **Evaluation Metrics:**
    - F1 Score (for class imbalance)
    - ROC-AUC Score (for discrimination power)

    **Interpretability:**
    - SHAP used to explain individual predictions and identify key churn drivers like `MonthlyCharges`, `Contract`, `Tenure`, and `CLV`.
    """)

st.markdown("This app predicts customer churn for SaaS or telecom users using LightGBM.")

# Load model
try:
    model = joblib.load("lgbm_model.joblib")
except:
    st.error("Model file 'lgbm_model.joblib' not found. Please add your trained model.")
    st.stop()

# Manual encodings
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
online_security_map = {"Yes": 1, "No": 0}
payment_method_map = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}

# Sidebar input
st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges ($)", 18.0, 130.0, 70.0)
contract = st.sidebar.selectbox("Contract Type", list(contract_map.keys()))
security = st.sidebar.selectbox("Online Security", list(online_security_map.keys()))
payment = st.sidebar.selectbox("Payment Method", list(payment_method_map.keys()))

# Prepare input
df_single = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "Contract": contract_map[contract],
    "OnlineSecurity": online_security_map[security],
    "PaymentMethod": payment_method_map[payment],
    "CLV": tenure * monthly
}])

# Predict
st.subheader("Single Prediction")
if st.button("Predict Churn"):
    prob = model.predict_proba(df_single)[0][1]
    st.metric("Churn Probability", f"{prob:.2%}")
    if prob > 0.7:
        st.warning("⚠️ High Risk of Churn")
    elif prob > 0.4:
        st.info("Moderate Risk")
    else:
        st.success("Low Risk")

# CSV Upload
st.markdown("---")
st.header("Batch Prediction from CSV")
uploaded = st.file_uploader("Upload a CSV with: tenure, MonthlyCharges, Contract, OnlineSecurity, PaymentMethod", type="csv")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        required_cols = ["tenure", "MonthlyCharges", "Contract", "OnlineSecurity", "PaymentMethod", "CLV"]
        if all(col in df.columns for col in required_cols):
            df = df[required_cols]
            preds = model.predict_proba(df)[:, 1]
            df["Churn_Probability"] = preds
            st.success("Prediction complete.")
            st.dataframe(df)
            st.download_button("Download Results", df.to_csv(index=False), file_name="churn_predictions.csv")
        else:
            st.error(f"Missing required columns. Expected: {required_cols}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a valid CSV file to run batch predictions.")


if uploaded:
    try:
        df = pd.read_csv(uploaded)
        df['Contract'] = df['Contract'].map(contract_map)
        df['OnlineSecurity'] = df['OnlineSecurity'].map(online_security_map)
        df['PaymentMethod'] = df['PaymentMethod'].map(payment_method_map)
        df['CLV'] = df['tenure'] * df['MonthlyCharges']
        df['Churn_Probability'] = model.predict_proba(df)[:, 1]
        st.dataframe(df)
        st.download_button("Download Results", data=df.to_csv(index=False), file_name="predictions.csv")
    except Exception as e:
        st.error(f"Error: {e}")
