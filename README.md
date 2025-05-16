# 🧠 Churn Reduction Strategy – SaaS Subscription Model

This project presents an end-to-end machine learning pipeline to predict customer churn using the IBM Telco dataset. It includes interpretability, bias handling, and strategic recommendations to improve retention in a subscription-based business.

## 🔍 Objective
- Predict customer churn using classification models
- Explain model behavior using SHAP
- Recommend retention strategies based on actionable insights

## 🗂️ Key Components
- Exploratory Data Analysis (EDA)
- Feature Engineering (CLV, ContractLength, Interactions)
- Bias Handling: SMOTEENN resampling
- Models: XGBoost, LightGBM, CatBoost, Ensemble
- Evaluation: ROC-AUC, F1 Score
- Model Explainability: SHAP
- Streamlit App for deployment

## 📈 Results
- Ensemble ROC-AUC Score: **0.84+**
- Top features influencing churn: `MonthlyCharges`, `Contract`, `Tenure`, `CLV`, `OnlineSecurity`

## 🧪 Technologies Used
- Python, Pandas, Scikit-learn
- XGBoost, LightGBM, CatBoost
- SHAP, SMOTEENN
- Streamlit



## 🚀 Live Project: Customer Churn Prediction (SaaS Model)

👉 **[Launch the App]((https://churn-prediction-saas-model-mumkwp86yanxpnqeh7mwkv.streamlit.app/))**

Use this app to:

* Predict the churn probability for individual customers based on contract and billing details
* Upload a CSV file for batch churn risk scoring
* See results instantly and download them for further action

Note: use the cleaned version of the csv file 


## 🛠 Run Locally

1. Clone the repo  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Sample CSV Format
```
tenure,MonthlyCharges,Contract,OnlineSecurity,PaymentMethod
12,75.9,Month-to-month,No,Bank transfer
```

## 📜 License
MIT
