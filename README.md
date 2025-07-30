```markdown
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

## Conclusion

The Quantum Neurological Disorder Diagnosis System demonstrates the potential of combining classical machine learning with quantum-inspired methods to aid in early and non-invasive diagnosis of neurological conditions. By leveraging voice-based data and providing an interactive, user-friendly interface, the system offers a scalable solution that bridges technology and healthcare. While currently based on simulated data and models, this project lays the groundwork for future advancements in clinical diagnostics, particularly in integrating real-world data and cutting-edge quantum algorithms.

## Prerequisites

Make sure Python is installed (3.7+ recommended). Then install the required libraries:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn pickle
```

## Running the App

1. Clone or download the repository.  
2. Ensure files like `app.py`, `models/qml_model.pkl`, and scripts in the `utils/` folder are present.  
3. Open terminal in the project directory and run:

```bash
streamlit run app.py
```

4. The app will open in your default browser.  
5. Upload the CSV file and receive predictions.  
6. Option to download the results is provided.

## Sample CSV Format

```csv
name,MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),...
phon_R01_S01_1,119.992,157.302,74.997,0.00784,...
phon_R01_S01_2,122.4,148.65,113.819,0.00968,...
```

> Make sure your CSV format matches the training data format used in the project.

## Future Enhancements

- Integration with real-world EEG/MEG or clinical voice datasets  
- Deployment of advanced quantum models like QNN or Q-SVM  
- Expansion to multi-class classification for different neurological disorders  
- Cloud-based deployment for remote clinical usage  
- Real-time voice recording and prediction functionality  
- Enhanced UI with interactive, dynamic visualizations

## Data Summary

- Features extracted from voice/acoustic signals  
- Numerical values like frequency bands, jitter, shimmer, and amplitude  
- Target label (status):  
  - `0` = No Disorder  
  - `1` = Neurological Disorder  
- Feature engineering includes statistical analysis like mean, variance, skewness, etc.

## Key Highlights

- Integration of quantum-inspired ML techniques in healthcare  
- Non-invasive, voice-based diagnosis system  
- User-friendly, interactive dashboard via Streamlit  
- Supports CSV upload, visualization, prediction, and result download  
- Designed with modular structure and real-world use cases in mind
```
