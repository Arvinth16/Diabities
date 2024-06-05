import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import xgboost as xgb

# Load the dataset
data = pd.read_csv('diabetes.csv')  # Change this to your actual path

# Replace zero values with NaN
cols_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero_values] = data[cols_with_zero_values].replace(0, np.nan)

# Use KNNImputer to fill missing values
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data)

# Convert back to DataFrame
data = pd.DataFrame(data_imputed, columns=data.columns)

# Feature Engineering: Add Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(data.drop(columns='Outcome'))
X_poly = pd.DataFrame(X_poly)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_poly)

# Separate features and label
y = data['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape the data for Conv1D
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the CNN model building function
def build_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv1D(128, 2, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

# Build the CNN model
cnn_model = build_cnn_model((X_train_cnn.shape[1], 1))

# Compile the model
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the CNN model
history = cnn_model.fit(X_train_cnn, y_train, epochs=300, batch_size=32, 
                        validation_data=(X_test_cnn, y_test), callbacks=[early_stopping, model_checkpoint], verbose=1)

# Evaluate the CNN model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test)
print(f'CNN Model Accuracy: {cnn_accuracy:.2f}')

# XGBoost Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Ensemble Model: Voting Classifier
# Note: Keras models can be integrated in a voting ensemble using custom wrappers if needed. 
# Here, we just show the CNN model evaluation. Implementing a full voting ensemble requires additional steps.
# ensemble_model = VotingClassifier(estimators=[
#     ('cnn', cnn_model),  # This requires a custom wrapper
#     ('xgb', xgb_model)
# ], voting='soft')

# # Train the ensemble model
# ensemble_model.fit(X_train, y_train)

# # Evaluate the ensemble model
# ensemble_accuracy = ensemble_model.score(X_test, y_test)
# print(f'Ensemble Model Accuracy: {ensemble_accuracy:.2f}')

# Plotting accuracy and loss for CNN model
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
