import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data (Target is: Dropout, Graduate, or Enrolled)
df = pd.read_csv('data.csv', sep=';')

# Convert Target: Dropout = 1 (Risk), others = 0 (Safe)
df['Target'] = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})

# Select features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Scale data between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# IMPORTANT: Reshape for CNN-LSTM (3D: Samples, Time-steps, Features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Save for the next step
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
print("Data Preparation Complete!")