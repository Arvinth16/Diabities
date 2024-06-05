from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the dataset (assume 'diabetes.csv' is the path to your dataset)
data = pd.read_csv('diabetes.csv')

# Replace zero values with NaN
cols_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero_values] = data[cols_with_zero_values].replace(0, np.nan)

# Use KNNImputer to fill missing values
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data.drop(columns='Outcome'))

# Convert back to DataFrame
data_imputed_df = pd.DataFrame(data_imputed, columns=data.columns[:-1])

# Feature Engineering: Add Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(data_imputed_df)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_poly)

# Load the trained model
cnn_model = load_model('best_model.keras')

# Define the preprocessing function
def preprocess_input(new_data, scaler, poly, imputer):
    new_data[cols_with_zero_values] = new_data[cols_with_zero_values].replace(0, np.nan)
    new_data_imputed = imputer.transform(new_data)
    X_poly = poly.transform(new_data_imputed)
    X_scaled = scaler.transform(X_poly)
    return X_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])
        
        user_data = pd.DataFrame({
            'Pregnancies': [Pregnancies],
            'Glucose': [Glucose],
            'BloodPressure': [BloodPressure],
            'SkinThickness': [SkinThickness],
            'Insulin': [Insulin],
            'BMI': [BMI],
            'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
            'Age': [Age]
        })
        
        X_new = preprocess_input(user_data, scaler, poly, imputer)
        X_new_cnn = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
        
        probability = cnn_model.predict(X_new_cnn)
        probability_value = probability[0][0]
        
        return render_template('result.html', probability=probability_value)

if __name__ == '__main__':
    app.run(debug=True)
