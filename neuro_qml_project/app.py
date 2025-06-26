# # app.py
# import streamlit as st
# import pandas as pd
# import pickle
# from utils.preprocessing import load_data
# from utils.visualization import plot_correlation_matrix, plot_class_distribution

# st.set_page_config(page_title='Neurological Disorder Diagnosis (Quantum)', layout='centered')
# st.title('🧠 Quantum Neurological Disorder Diagnosis System')

# uploaded_file = st.file_uploader('Upload your CSV file (with same format)', type='csv')

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.success('File uploaded successfully!')
#     st.dataframe(df.head())

#     if st.button('Show Exploratory Data Analysis'):
#         st.subheader('Class Distribution')
#         st.pyplot(plot_class_distribution(df))

#         st.subheader('Correlation Matrix')
#         st.pyplot(plot_correlation_matrix(df))

#     st.subheader('Prediction Result (Quantum Classifier)')

#     with open('models/qml_model.pkl', 'rb') as f:
#         model, scaler = pickle.load(f)


#     X_new = df.drop(['name', 'status'], axis=1)
#     X_scaled_new = scaler.transform(X_new)

#     # Reduce to 4 features
#     X_reduced_new = X_scaled_new[:, :4]
#     predictions = model.predict(X_reduced_new)

#     df['Prediction'] = predictions
#     df['Prediction_Label'] = df['Prediction'].map({0: 'No Disorder', 1: 'Neurological Disorder'})
#     st.dataframe(df[['name', 'Prediction_Label']])

#     st.download_button('Download Prediction Results as CSV', data=df.to_csv(index=False).encode(), file_name='prediction_results.csv')
# else:
#     st.info('Please upload a CSV file to get started.')
import streamlit as st
import pandas as pd
import pickle
from utils.preprocessing import load_data
from utils.visualization import plot_correlation_matrix, plot_class_distribution

st.set_page_config(page_title='Neurological Disorder Diagnosis (Quantum)', layout='centered')
st.title('🧠 Quantum Neurological Disorder Diagnosis System')

uploaded_file = st.file_uploader('Upload your CSV file (with same format)', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success('File uploaded successfully!')
    st.dataframe(df.head())

    if st.button('Show Exploratory Data Analysis'):
        st.subheader('Class Distribution')
        st.pyplot(plot_class_distribution(df))

        st.subheader('Correlation Matrix')
        st.pyplot(plot_correlation_matrix(df))

    st.subheader('Prediction Result (Quantum Classifier)')

    with open('models/qml_model.pkl', 'rb') as f:
        model, scaler = pickle.load(f)

    X_new = df.drop(['name', 'status'], axis=1)  # Make sure you keep all features except 'name' and 'status'
    X_scaled_new = scaler.transform(X_new)  # Scale the input features the same way as during training

    # Make predictions using all the features
    predictions = model.predict(X_scaled_new)

    # Map predictions to labels
    df['Prediction'] = predictions
    df['Prediction_Label'] = df['Prediction'].map({0: 'No Disorder', 1: 'Neurological Disorder'})

    # Display the prediction result in a readable format
    st.dataframe(df[['name', 'Prediction_Label']])

    # Display detailed message for each person
    for index, row in df.iterrows():
        if row['Prediction'] == 1:
            # Here you can customize the disorder types based on your model's output
            st.write(f"{row['name']} has a neurological disorder. Based on the model's prediction, "
                     "the disorder could be related to brain activity abnormalities.")
        else:
            st.write(f"{row['name']} does not have any neurological disorder.")

    # Option to download the results
    st.download_button('Download Prediction Results as CSV', data=df.to_csv(index=False).encode(), file_name='prediction_results.csv')

else:
    st.info('Please upload a CSV file to get started.')
