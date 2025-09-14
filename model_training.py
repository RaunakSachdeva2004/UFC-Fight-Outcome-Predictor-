import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# --- 1. Data Loading and Initial Preparation ---
# Load the dataset
try:
    data = pd.read_csv('ufc-master.csv')
except FileNotFoundError:
    print("Error: 'ufc-master.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the correct directory.")
    exit()

# Select relevant features for a simple model
# We'll use basic physical attributes and fight records.
features = [
    'R_Height_cms', 'R_Weight_lbs', 'R_Reach_cms', 'R_wins', 'R_losses',
    'B_Height_cms', 'B_Weight_lbs', 'B_Reach_cms', 'B_wins', 'B_losses',
    'R_Stance', 'B_Stance', 'Winner'
]
df = data[features].copy()

# Drop rows where the winner is a draw or no contest, as we are doing binary classification
df = df[df['Winner'].isin(['Red', 'Blue'])]
df.dropna(subset=['Winner'], inplace=True)


# --- 2. Feature Engineering and Preprocessing ---
# Separate features (X) and target (y)
X = df.drop('Winner', axis=1)
y = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0) # Red: 1, Blue: 0

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Create preprocessing pipelines for numerical and categorical data
# For numerical data, we'll impute missing values with the mean and then scale them.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# For categorical data, we'll impute missing values with the most frequent value and then one-hot encode them.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# --- 3. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 4. Model Building (Neural Network) ---
# Apply the preprocessing pipeline to the training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train_processed.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
print("Model Summary:")
model.summary()


# --- 5. Model Training ---
print("\nStarting model training...")
history = model.fit(
    X_train_processed,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
print("Model training finished.")


# --- 6. Model Evaluation ---
print("\nEvaluating model performance...")
loss, accuracy = model.evaluate(X_test_processed, y_test)
print(f'\nTest Accuracy: {accuracy:.4f}')
print(f'Test Loss: {loss:.4f}')


# --- 7. Saving the Model and Preprocessor ---
# Save the trained model
model.save('ufc_prediction_model.h5')
# Save the preprocessor object (which includes the scaler)
joblib.dump(preprocessor, 'preprocessor.joblib')

print("\nModel ('ufc_prediction_model.h5') and preprocessor ('preprocessor.joblib') have been saved successfully!")
