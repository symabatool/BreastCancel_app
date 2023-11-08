import streamlit as st
import pandas as pd
import joblib 
import time

with st.spinner('Fetching Latest ML Model'):
    time.sleep(1)
    st.success('Model Loaded!')
st.title('Breast Cancer Diagnosis Predictor')


radius_mean = st.number_input('Radius Mean', min_value=0.0, value=13.0)
texture_mean = st.number_input('Texture Mean', min_value=0.0, value=15.0)
perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, value=100.0)
area_mean = st.number_input('Area Mean', min_value=0.0, value=500.0)
smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, value=0.1)
compactness_mean = st.number_input('Compactness Mean', min_value=0.0, value=0.2)
concavity_mean = st.number_input('Concavity Mean', min_value=0.0, value=0.3)
concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, value=0.04)


if st.button('Predict Diagnosis'):
    input_data = [radius_mean, texture_mean,perimeter_mean,	area_mean,	smoothness_mean,	compactness_mean,	concavity_mean,	concave_points_mean	]  # Add all input values
    BreastCancer=joblib.load('breast_cancer_model.pkl')
    prediction = BreastCancer.predict([input_data])[0]
    st.write(f'Predicted Diagnosis: {prediction}')
