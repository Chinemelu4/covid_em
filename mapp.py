
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from pycaret.classification import *
from imblearn.over_sampling import SMOTE

st.write("""
# Emergency Department Length of Stay Prediction for COVID-19 Patients
This app predicts the length of stay of COVID-19 Patients!
""")

st.sidebar.header('User Input Parameters')


def user_input_features():
    Bed_vailable= st.sidebar.selectbox('Bed Available',('Yes','No'))
    if Bed_vailable=='Yes':
        Bed_vailable=1
    else:
        Bed_vailable=0 
        
    Physician_vailable= st.sidebar.selectbox('Physician Available',('Yes','No'))
    if Physician_vailable=='Yes':
        Physician_vailable=1
    else:
        Physician_vailable=0
        
    Nurse_vailable= st.sidebar.selectbox('Nurse Available',('Yes','No'))
    if Nurse_vailable=='Yes':
        Nurse_vailable=1
    else:
        Nurse_vailable=0
        
    HAS_OBESITY= st.sidebar.selectbox('Obesity',('Yes','No'))
    if HAS_OBESITY=='Yes':
        HAS_OBESITY=1
    else:
        HAS_OBESITY=0
        
    HAS_CKD= st.sidebar.selectbox('CKD',('Yes','No'))
    if HAS_CKD=='Yes':
        HAS_CKD=1
    else:
        HAS_CKD=0
        
    HAS_CVD= st.sidebar.selectbox('CVD',('Yes','No'))
    if HAS_CVD=='Yes':
        HAS_CVD=1
    else:
        HAS_CVD=0
        
    
        
    Temp = st.number_input('Temperature',min_value=37)
    RR = st.number_input('Respiratory rate',min_value=1)
    SpO2 = st.number_input('Percent oxygen (%)',min_value=18)
    ESI_x = st.number_input('Severity index',min_value=1)
    data = {'Temp': Temp,
            'RR':  RR ,
            'SpO2':SpO2,
            'ESI_x': ESI_x,
            'Bed Available':Bed_vailable,
            'Physician Available':Physician_vailable,
            'Nurse Available':Nurse_vailable,
            'HAS_OBESITY':HAS_OBESITY,
            'HAS_CKD':HAS_CKD,
            'HAS_CVD':HAS_CVD}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


model=load_model('gbc_deployment_13Jan2022_v3')


if st.button('PREDICT'):
    classes=['Short stay','Long Stay']
    y_out=predict_model(model, data = input_df)
    y_out1=y_out['Label'][0]
    if y_out1==1:
        st.write('This patient will stay for over 4hrs')
    else:
        st.write('This patient will stay for less than 4 hrs')
        
