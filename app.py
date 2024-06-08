import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('xgboost_model.pkl')

# Define the prediction function
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

# Define the Streamlit app
def main():
    st.title("Bank Customer Churn Prediction")

    # Input fields
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    country = st.selectbox("Country", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance", value=10000.0)
    products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    active_member = st.selectbox("Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", value=50000.0)

    # Encoding input features
    country_map = {'France': 0, 'Spain': 1, 'Germany': 2}
    gender_map = {'Male': 0, 'Female': 1}
    credit_card_map = {'Yes': 1, 'No': 0}
    active_member_map = {'Yes': 1, 'No': 0}

    input_data = pd.DataFrame({
        'credit_score': [credit_score],
        'country': [country_map[country]],
        'gender': [gender_map[gender]],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_card_map[credit_card]],
        'active_member': [active_member_map[active_member]],
        'estimated_salary': [estimated_salary]
    })

    # Predict
    if st.button("Predict"):
        prediction = predict_churn(input_data)
        if prediction[0] == 1:
            st.warning("The customer is likely to churn.")
        else:
            st.success("The customer is not likely to churn.")

if __name__ == '__main__':
    main()