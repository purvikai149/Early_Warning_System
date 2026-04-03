import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout

# Load prepared data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Build Hybrid Model
model = Sequential([
    Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(1, X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # Binary output: 0 or 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Save the trained 'brain'
model.save('student_model.h5')
print("Model Training Complete and Saved!")
import shap
import matplotlib.pyplot as plt

# 1. Use a subset of training data as a background for SHAP
# SHAP explains the difference between the average prediction and the current one
background = X_train[:100] 
explainer = shap.GradientExplainer(model, background)

# 2. Calculate SHAP values for 10 test samples
shap_values = explainer.shap_values(X_test[:10])

# 3. Create the summary plot
# This identifies which features (Grades, Scholarship, etc.) are most influential
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[0][:, 0, :], features=X.columns, show=False)
plt.savefig('shap_importance.png') # This saves the image for your report
print("✅ SHAP plot saved as 'shap_importance.png'")