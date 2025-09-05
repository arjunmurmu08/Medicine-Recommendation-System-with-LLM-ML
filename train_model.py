import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os

# Step 1: Load the training data
train_path = os.path.join('backend', 'datasets', 'Training.csv')
df = pd.read_csv(train_path)

# Step 2: Split into features and label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Step 3: Initialize and train the SVM model
svc = SVC(probability=True)
svc.fit(X, y)

# Step 4: Ensure the models folder exists
os.makedirs('models', exist_ok=True)

# Step 5: Save the trained model to models/svc.pkl
model_path = os.path.join('models', 'svc.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(svc, f)

print("✅ Model has been trained and saved to models/svc.pkl")
