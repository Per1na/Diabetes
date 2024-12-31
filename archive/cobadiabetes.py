import pickle
import streamlit as st

# Membaca Model
diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

# Judul Web
st.title('Data Mining Prediksi Diabetes')

# Membagi Kolom
col1, col2 = st.columns(2)

with col1 :
    Pregnancies = st.number_input('Input Nilai Pregnancies')

with col2 :
    Glucose = st.number_input('Input Nilai Glucose')

with col1 :
    BloodPressure = st.number_input('Input Nilai Blood Pressure')

with col2 :
    SkinThickness = st.number_input('Input Nilai Skin Thickness')

with col1 :
    Insulin = st.number_input('input Nilai Insulin')

with col2 :
    BMI = st.number_input('Input Nilai BMI')

with col1 :
    DiabetesPedigreeFunction = st.number_input('Input Nilai Diabetes Pedigree Function')

with col2 :
    Age = st.number_input('Input Nilai Age')

# Code untuk Prediksi
diab_diagnosis = ''

# Membuat Tombol untuk Prediksi
if st.button('Test Prediksi Diabetes') :
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    if(diab_prediction[0] == 1):
        diab_diagnosis = 'Pasien terkena Diabetes'
    else :
        diab_diagnosis = 'Pasien tidak terkena Diabetes'

    st.success(diab_diagnosis)