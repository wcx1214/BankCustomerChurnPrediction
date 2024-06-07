from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = joblib.load('xgboost_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    credit_score = float(request.form['credit_score'])
    country = int(request.form['country'])
    gender = int(request.form['gender'])
    age = float(request.form['age'])
    tenure = float(request.form['tenure'])
    balance = float(request.form['balance'])
    products_number = int(request.form['products_number'])
    credit_card = int(request.form['credit_card'])
    active_member = int(request.form['active_member'])
    estimated_salary = float(request.form['estimated_salary'])
    

    # Create the feature array
    features = np.array([[credit_score, country, gender, age, tenure, balance, products_number,
                          credit_card, active_member, estimated_salary]])

    # Standard scaling using the same scaler used during training
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Make a prediction
    prediction = model.predict(scaled_features)[0]
    prediction_text = 'Churn' if prediction == 1 else 'No Churn'
    
    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
