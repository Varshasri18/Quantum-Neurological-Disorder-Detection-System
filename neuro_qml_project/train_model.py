import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load your dataset (Make sure it is in the same format as in your app)
df = pd.read_csv("data/sample.csv")

# Example preprocessing
X = df.drop(['name', 'status'], axis=1)  # features
y = df['status']  # target

# Train a model (example: Random Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save the trained model and scaler
with open('models/qml_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)

print("Model and scaler saved successfully.")
