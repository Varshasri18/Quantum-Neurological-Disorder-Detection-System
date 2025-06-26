# train_qml.py

import pandas as pd
from utils.preprocessing import load_data, preprocess_data, split_data

# Quantum ML
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import joblib

# 1. Load data
df = load_data('data/sample.csv')

# 2. Preprocess data
X_scaled, y = preprocess_data(df)

# 3. Split data
X_train, X_test, y_train, y_test = split_data(X_scaled, y)

# 4. Train classical model (as baseline)
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Classical Model Accuracy: {acc*100:.2f}%")

# 6. Save model
joblib.dump(clf, 'models/classical_model.joblib')
print("Model saved to models/classical_model.joblib")
