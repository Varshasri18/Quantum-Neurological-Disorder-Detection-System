# utils/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Loads CSV data"""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Splits features/labels and scales features"""
    X = df.drop(columns=['name', 'status'])
    y = df['status']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits into train/test"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
