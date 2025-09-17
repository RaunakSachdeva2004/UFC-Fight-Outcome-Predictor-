import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# --- 1. Data Loading and Preparation ---
print("--- Step 1: Loading and Preparing Data ---")
try:
    data = pd.read_csv('ufc-master.csv')
    print("Dataset 'ufc-master.csv' loaded successfully.")
except FileNotFoundError:
    print("\n[ERROR] 'ufc-master.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same folder as this script.")
    exit()

features = [
    'R_Height_cms', 'R_Weight_lbs', 'R_Reach_cms', 'R_wins', 'R_losses',
    'B_Height_cms', 'B_Weight_lbs', 'B_Reach_cms', 'B_wins', 'B_losses',
    'R_Stance', 'B_Stance', 'Winner'
]
df = data[features].copy()
df = df[df['Winner'].isin(['Red', 'Blue'])]
df.dropna(subset=['Winner'], inplace=True)

X = df.drop('Winner', axis=1)
y = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)
print("Data prepared for preprocessing.")

# --- 2. Preprocessing Pipeline ---
print("\n--- Step 2: Building Preprocessing Pipeline ---")
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # FIX: Force the output to be a dense numpy array instead of a sparse matrix
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)])
print("Preprocessing pipeline created.")

# --- 3. Data Splitting and Transformation ---
print("\n--- Step 3: Splitting and Transforming Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor and transform the data. No .toarray() needed now.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
print("Data successfully converted to PyTorch tensors.")

# --- 4. Model Definition ---
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu2(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output_layer(x))
        return x

input_features = X_train_tensor.shape[1]
model = NeuralNet(input_size=input_features)
print("\n--- Step 4: Model Summary ---")
print(model)

# --- 5. Training the Model ---
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50
batch_size = 32

print(f"\n--- Step 5: Starting Model Training ({epochs} epochs) ---")
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
print("Model training finished.")

# --- 6. Evaluating the Model ---
print("\n--- Step 6: Evaluating Model Performance ---")
model.eval()
with torch.no_grad():
    y_predicted = model(X_test_tensor)
    test_loss = criterion(y_predicted, y_test_tensor)
    predicted_class = (y_predicted > 0.5).float()
    accuracy = (predicted_class == y_test_tensor).float().mean()
    print(f'Test Accuracy: {accuracy.item():.4f}')
    print(f'Test Loss: {test_loss.item():.4f}')

# --- 7. Saving Files ---
print("\n--- Step 7: Saving Model and Preprocessor ---")
model_path = 'ufc_prediction_model.pth'
preprocessor_path = 'preprocessor.joblib'
torch.save(model.state_dict(), model_path)
joblib.dump(preprocessor, preprocessor_path)

print(f"\n[SUCCESS] Model saved to '{os.path.abspath(model_path)}'")
print(f"[SUCCESS] Preprocessor saved to '{os.path.abspath(preprocessor_path)}'")

