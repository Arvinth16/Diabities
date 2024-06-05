import tensorflow.keras.models as keras  # For TensorFlow 2.x and later
# For TensorFlow 1.x, use: from keras.models import keras

from sklearn.preprocessing import MinMaxScaler  # Assuming MinMaxScaler was used

def get_user_input():
  """
  Prompts the user for input and performs validation for expected data types and ranges.

  Returns:
    A list containing the validated user-provided values for the 9 attributes.
  """
  while True:
    try:
      pregnancies = int(input("Enter number of pregnancies (must be a whole number): "))
      if pregnancies < 0:
        print("Invalid input: Number of pregnancies cannot be negative.")
        continue

      glucose = float(input("Enter glucose level (mg/dL): "))
      # Add checks for reasonable glucose level range (e.g., 0-200)
      if glucose < 0:
        print("Invalid input: Glucose level cannot be negative.")
        continue

      blood_pressure = int(input("Enter blood pressure (mmHg): "))
      # Add checks for reasonable blood pressure range

      skin_thickness = float(input("Enter skin thickness (mm): "))
      # Add checks for reasonable skin thickness range (if known)

      insulin = float(input("Enter insulin level (uU/mL) (0 if not applicable): "))
      # Handle negative values (if applicable)

      bmi = float(input("Enter body mass index (kg/m²): "))
      # Add checks for reasonable BMI range

      diabetes_pedigree_function = float(input("Enter diabetes pedigree function: "))
      # Add checks for reasonable range (if known)

      age = int(input("Enter age (years): "))
      if age < 0:
        print("Invalid input: Age cannot be negative.")
        continue

      return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, 0]
    except ValueError:
      print("Invalid input. Please enter numbers only where applicable.")

# Check TensorFlow version to determine the appropriate import for load_model
try:
  # For TensorFlow 2.x and later
  from tensorflow.keras.models import load_model
except ImportError:
  # For TensorFlow 1.x (assuming keras as submodule)
  from keras.models import load_model

# Load the saved CNN model (assuming 'best_model.keras' is the filename)
try:
  model = load_model('best_model.keras')
except OSError:
  print("Error: Could not load model 'best_model.keras'. Please check the filename and path.")

# Get user input
new_data = get_user_input()

# Preprocess the data if necessary (assuming MinMaxScaler was used during training)
scaler = MinMaxScaler()  # Assuming MinMaxScaler was used for scaling
scaled_data = scaler.fit_transform(new_data)

# Make prediction
prediction = model.predict(scaled_data)
probability = prediction[0][0]  # Assuming the output is a single probability value

# Print the results
print(f"Probability of having diabetes: {probability:.2f}")

# Interpretation (Optional)
risk_levels = {
    0: "Low Risk",
    1: "Moderate Risk (Consider consulting a doctor)",
    2: "High Risk (See a doctor for evaluation)"
}
risk_category = risk_levels.get(int(probability > 0.7) + int(probability > 0.5))
print(f"Risk Category: {risk_category}")

# Note: Adjust risk thresholds and interpretation as needed. Consult a doctor for diagnosis.
