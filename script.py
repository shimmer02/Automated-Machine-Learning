import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pycaret.classification as classification
import pycaret.regression as regression
import pycaret.clustering as clustering

# Set Streamlit page config
st.set_page_config(page_title='AutoML App', page_icon=':robot:', layout='wide')

# Sidebar Navigation
with st.sidebar:
    st.image('icon.png', width=150)
    st.title('Automated ML')
    choice = st.selectbox("Navigation", ['Upload Data', 'Data Preprocessing', 'Data Analysis', 'Regression Modelling',
                                          'Classification Modelling', 'Clustering',
                                          'Regressor Testing', 'Classifier Testing', 'Clustering Testing'])
    st.info('An AI-driven AutoML web app for easy machine learning model training and testing.')

# Global dataframe
if os.path.exists('original_data.csv'):
    df = pd.read_csv('original_data.csv')
else:
    df = None

# Upload Data
if choice == 'Upload Data':
    st.title('Upload Your Dataset')
    file = st.file_uploader('Upload a CSV file', type=['csv'])
    if file:
        df = pd.read_csv(file)
        df.to_csv('original_data.csv', index=False)
        st.success('File uploaded successfully!')
        st.dataframe(df)
    else:
        st.warning('No file uploaded yet.')

# Data Preprocessing
if choice == 'Data Preprocessing':
    st.title('Data Preprocessing')
    if df is not None:
        st.subheader("Handle Missing Values")
        missing_values_option = st.radio("Select how to handle missing values:", ['Drop Rows', 'Fill with Mean', 'Fill with Median', 'Fill with Mode'])
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if missing_values_option == 'Drop Rows':
            df.dropna(inplace=True)
        elif missing_values_option == 'Fill with Mean':
            df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
            df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
        elif missing_values_option == 'Fill with Median':
            df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
            df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
        elif missing_values_option == 'Fill with Mode':
            for col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        st.success('Missing values handled successfully!')
        
        st.subheader("Feature Scaling")
        scale_option = st.radio("Select a scaling method:", ['None', 'Standardization (Z-score)', 'Min-Max Scaling'])
        
        if scale_option != 'None':
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            
            if scale_option == 'Standardization (Z-score)':
                scaler = StandardScaler()
            elif scale_option == 'Min-Max Scaling':
                scaler = MinMaxScaler()
            
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            st.success('Feature scaling applied successfully!')
        
        st.subheader("Encode Categorical Variables")
        if categorical_columns:
            encoding_method = st.radio("Select encoding method:", ['One-Hot Encoding', 'Label Encoding'])
            from sklearn.preprocessing import LabelEncoder
            if encoding_method == 'Label Encoding':
                encoder = LabelEncoder()
                for col in categorical_columns:
                    df[col] = encoder.fit_transform(df[col].astype(str))
            else:
                df = pd.get_dummies(df, columns=categorical_columns)
            st.success('Categorical encoding applied successfully!')
        
        # Save preprocessed data
        df.to_csv('preprocessed_data.csv', index=False)
        st.dataframe(df)
        st.download_button("Download Preprocessed Data", df.to_csv(index=False).encode('utf-8'), "preprocessed_data.csv", "text/csv")
    else:
        st.error('Please upload a dataset first.')
#data analysis
if choice == 'Data Analysis':
    st.title('Exploratory Data Analysis')
    report = ProfileReport(df)
    st_profile_report(report)

# Regression Modelling
if choice == 'Regression Modelling':
    st.title('Train a Regression Model')
    if df is not None:
        target = st.selectbox('Select Target Parameter', df.columns)
        if st.button('Train Model'):
            with st.spinner('Training regression models...'):
                regression.setup(df, target=target)
                models_comparison = regression.compare_models(n_select=5, sort='R2', turbo=True)
              
                st.subheader("Model Comparison")
                st.dataframe(regression.pull())  

                best_model = models_comparison[0]
                tuned_model = regression.tune_model(best_model)
                regression.save_model(tuned_model, 'best_regressor')

                st.success('Model training completed!')
                with open('best_regressor.pkl', 'rb') as f:
                    st.download_button('Download Model', f, 'trained_regressor.pkl')
    else:
        st.error('Please upload a dataset first.')


# Classification Modelling
if choice == 'Classification Modelling':
    st.title('Train a Classification Model')
    if df is not None:
        target = st.selectbox('Select Target Parameter', df.columns)
        if st.button('Train Model'):
            with st.spinner('Training classification models...'):
                classification.setup(df, target=target)
                models_comparison = classification.compare_models(n_select=5, sort='Accuracy', turbo=True)
                
                # Display comparison table
                st.subheader("Model Comparison")
                st.dataframe(classification.pull())  

                # Allow user to select best model
                best_model = models_comparison[0]
                tuned_model = classification.tune_model(best_model)
                classification.save_model(tuned_model, 'best_classifier')

                st.success('Model training completed!')
                with open('best_classifier.pkl', 'rb') as f:
                    st.download_button('Download Model', f, 'trained_classifier.pkl')
    else:
        st.error('Please upload a dataset first.')

# Clustering
if choice == 'Clustering':
    st.title('Train a Clustering Model')
    if df is not None:
        if st.button('Train Clustering Model'):
            with st.spinner('Training clustering model...'):
                clustering.setup(df)
                best_model = clustering.create_model('kmeans')
                clustering.save_model(best_model, 'best_clustering')
                st.success('Model training completed!')
                with open('best_clustering.pkl', 'rb') as f:
                    st.download_button('Download Model', f, 'trained_clustering.pkl')
    else:
        st.error('Please upload a dataset first.')

# Model Testing Sections
if choice == 'Regressor Testing':
    st.title('Test a Regression Model')
    test_file = st.file_uploader('Upload Test CSV for Regression', type=['csv'])
    if test_file is not None:
        test_df = pd.read_csv(test_file)
        pipeline = regression.load_model('best_regressor')
        predictions = regression.predict_model(pipeline, data=test_df)
        st.dataframe(predictions)

if choice == 'Classifier Testing':
    st.title('Test a Classification Model')
    test_file = st.file_uploader('Upload Test CSV for Classification', type=['csv'])
    if test_file is not None:
        test_df = pd.read_csv(test_file)
        pipeline = classification.load_model('best_classifier')
        predictions = classification.predict_model(pipeline, data=test_df)
        st.dataframe(predictions)

if choice == 'Clustering Testing':
    st.title('Test a Clustering Model')
    test_file = st.file_uploader('Upload Test CSV for Clustering', type=['csv'])
    if test_file is not None:
        test_df = pd.read_csv(test_file)
        pipeline = clustering.load_model('best_clustering')
        predictions = clustering.predict_model(pipeline, data=test_df)
        st.dataframe(predictions)

