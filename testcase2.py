import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from tensorflow.keras.models import load_model

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

# Separate features and label
y = data['Outcome']

# Define the preprocessing function
def preprocess_input(new_data, scaler, poly, imputer):
    new_data[cols_with_zero_values] = new_data[cols_with_zero_values].replace(0, np.nan)
    new_data_imputed = imputer.transform(new_data)
    X_poly = poly.transform(new_data_imputed)
    X_scaled = scaler.transform(X_poly)
    return X_scaled

# Load the trained model
cnn_model = load_model('best_model.keras')

# Function to get user input
def get_user_input():
    print("Enter the following details:")
    Pregnancies = int(input("Pregnancies: "))
    Glucose = float(input("Glucose: "))
    BloodPressure = float(input("BloodPressure: "))
    SkinThickness = float(input("SkinThickness: "))
    Insulin = float(input("Insulin: "))
    BMI = float(input("BMI: "))
    DiabetesPedigreeFunction = float(input("DiabetesPedigreeFunction: "))
    Age = int(input("Age: "))
    
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
    
    return user_data

# Get user input
user_data = get_user_input()
print("User Input Data:")
print(user_data)

# Preprocess the input data
X_new = preprocess_input(user_data, scaler, poly, imputer)
print("Preprocessed Input Data:")
print(X_new)

# Reshape for the CNN model
X_new_cnn = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
print("Reshaped Input Data:")
print(X_new_cnn)

# Predict the probability
probability = cnn_model.predict(X_new_cnn)
print("Prediction Probability:")
print(probability)

print(f'The probability of having diabetes is: {probability[0][0]:.2f}')
