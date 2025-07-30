# Quantum Neurological Disorder Diagnosis System

A machine learning-based application developed using Python and Streamlit for early detection of neurological disorders based on voice data. This project, built under the Computer Science & Engineering department at MVJ College of Engineering, integrates classical machine learning and explores quantum-enhanced techniques to improve diagnostic accuracy through acoustic analysis.

## Overview

The system is designed to classify individuals as having a neurological disorder or not based on features derived from their voice recordings. Users can upload CSV files containing voice-related data, view exploratory analysis, and receive model predictions with the option to download results. The solution aims to assist in early, non-invasive screening of disorders like Parkinson’s using acoustic signal processing and machine learning.

## Technologies Used

- Python  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Streamlit (for web interface)  
- Random Forest Classifier  
- Quantum Machine Learning (simulated enhancements)

## How It Works

### Input
- Users upload a CSV file containing voice features such as jitter, shimmer, and frequency values.

### Preprocessing
- Unnecessary columns like `name` and `status` are dropped.
- Data is cleaned and scaled using a standard scaler.
- Preprocessed data is prepared for prediction.

### Prediction
- The processed data is passed to a trained Random Forest classifier enhanced with quantum-inspired techniques.
- The system predicts neurological status as either **“Neurological Disorder”** or **“No Disorder”**.

### Output
- Predicted results are displayed on the dashboard.
- Users can download the output as a CSV file.
- Visual insights such as correlation matrix and class distribution are also provided.

## Algorithms Used

### Random Forest Classifier
- Ensemble learning model that combines multiple decision trees.
- Offers high accuracy and robustness for binary classification.
- Used here to classify neurological status based on voice data.

### Quantum Machine Learning (QML)
- Simulated integration of quantum algorithms to explore speed and accuracy improvements.
- Demonstrates potential of quantum computing in medical diagnosis applications.

## Methodology

1. **Data Collection**  
   Voice data in CSV format with acoustic features like frequency, jitter, shimmer, etc.

2. **Data Preprocessing**  
   Removal of irrelevant features.  
   Feature scaling using a standard scaler.  
   Cleaned data is passed to the model.

3. **Model Training**  
   Trained Random Forest classifier using labeled data.  
   Quantum-enhanced modifications were explored for experimental performance improvements.

4. **Prediction**  
   Trained model is used for predicting unseen data.  
   Outputs are mapped to clear labels: **"Neurological Disorder"** or **"No Disorder"**.

5. **User Interaction**  
   Users upload a CSV file on a Streamlit dashboard.  
   Receive instant results and download predictions.

## Visualizations

- **Correlation Matrix**: Shows relationships between features to identify redundancy or multicollinearity.
- **Class Distribution**: Displays the balance between classes (e.g., Disorder vs No Disorder).
- **Confusion Matrix**: Gives a breakdown of true positives, false positives, true negatives, and false negatives.
- **Performance Metrics**: Includes Accuracy, Precision, Recall, and F1 Score.

## Prerequisites

Make sure Python is installed (3.7+ recommended). Then install the required libraries:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn pickle

## Conclusion

The Quantum Neurological Disorder Diagnosis System demonstrates the potential of combining classical machine learning with quantum-inspired methods to aid in early and non-invasive diagnosis of neurological conditions. By leveraging voice-based data and providing an interactive, user-friendly interface, the system offers a scalable solution that bridges technology and healthcare. While currently based on simulated data and models, this project lays the groundwork for future advancements in clinical diagnostics, particularly in integrating real-world data and cutting-edge quantum algorithms.
