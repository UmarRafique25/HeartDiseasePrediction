import streamlit as st
import numpy as np
import joblib

model = joblib.load('models/model.pkl')

scaler = joblib.load('models/scaler.pkl')

st.title('Heart Disease Prediction System')

age = st.number_input('Age')

trestbps = st.number_input('Blood Pressure')

chol = st.number_input('Cholesterol')

thalch = st.number_input('Max Heart Rate')

oldpeak = st.number_input('Old Peak')

ca = st.number_input('CA')

sex = st.selectbox('Sex', [0,1])

cp = st.selectbox('Chest Pain Type', [0,1,2,3])

fbs = st.selectbox('FBS', [0,1])

restecg = st.selectbox('Rest ECG', [0,1,2])

exang = st.selectbox('Exercise Angina', [0,1])

slope = st.selectbox('Slope', [0,1,2])

thal = st.selectbox('Thal', [0,1,2])

if st.button('Predict'):

    features = np.array([[

        age, sex, cp, trestbps, chol,
        fbs, restecg, thalch,
        exang, oldpeak, slope,
        ca, thal

    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error('High Risk of Heart Disease')
    else:
        st.success('Low Risk of Heart Disease')