import streamlit as st
import pandas as pd
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model


with st.sidebar:
    st.image('icon.png')
    st.title('Automated ML')
    choice = st.radio("Navigation", ['Upload', 'Data Analysis', 'Model Creation', 'Download Model', 'Test Model Predictions'])
    st.info('An AI-driven AutoML web app that automates the process of building, training, and deploying machine learning models, making data science accessible to everyone.')

if os.path.exists('original_data.csv'):
    df = pd.read_csv('original_data.csv', index_col=None)
if choice == 'Upload':
    file = st.file_uploader('Upload your Data here')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('original_data.csv', index =None)
        st.dataframe(df)

if choice == 'Data Analysis':
    st.title('Exploratiory Data Analysis')
    report = ProfileReport(df)
    st_profile_report(report)

if choice == 'Model Creation':
    st.title('Generation of ML model')
    target = st.selectbox('Select Target Parameter', df.columns)
    if st.button('Train Model'):
        setup(df, target=target)
        setup_df = pull()
        st.info('Model Experimentation Setiings')
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info('Generated Model')
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')

if choice == 'Download Model':
    st.title('Download Generated Model')
    st.info('You Can Download the generated model from here')
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download the Model', f, 'trained_model.pkl')

if choice == 'Test Model Predictions':
    st.title('Model Testing')
    st.info('Upload your test dataset')
    test = st.file_uploader('Upload Test csv file')
    df2 = pd.read_csv(test)
    if test: 
        st.info('This is your test dataset')
        st.dataframe(df2)
    pipeline = load_model('trained_model')
    st.info('This is the dataframe with the prediction labels')
    st.dataframe(predict_model(pipeline, df2))
