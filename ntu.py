import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv('./diabetes.csv')  # Ensure the file path is correct

# Replace zero values with NaN for specific columns
cols_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero_values] = data[cols_with_zero_values].replace(0, np.nan)

# Use KNNImputer to fill missing values
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data)

# Convert back to DataFrame
data = pd.DataFrame(data_imputed, columns=data.columns)

# Separate features and labels
X = data.drop(columns='Outcome')
y = data['Outcome']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

# Recommendation function
def recommend(prediction):
    if prediction == 1:
        medication = "Consult with your doctor for appropriate medication. Possible options include Metformin, Insulin."
        food_intake = "Follow a low-carb diet. Include more vegetables, lean proteins, and whole grains. Avoid sugary foods and beverages."
    else:
        medication = "No medication needed. Maintain a healthy lifestyle."
        food_intake = "Maintain a balanced diet rich in vegetables, fruits, whole grains, and lean proteins. Exercise regularly."
    return medication, food_intake

# Example of using the recommendation function
for i in range(5):  # Display recommendations for first 5 test samples
    pred = model.predict([X_test[i]])[0]
    medication, food_intake = recommend(pred)
    print(f'Prediction: {pred}, Medication: {medication}, Food Intake: {food_intake}')
