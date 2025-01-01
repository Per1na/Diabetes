import streamlit as st
import pickle
import pandas as pd
import numpy as np  # Tambahkan numpy di sini

# Load Model
MODEL_PATH = 'diabetes_model.sav'
with open(MODEL_PATH, 'rb') as file:
    diabetes_model = pickle.load(file)

# Streamlit App
st.title("Prediksi Diabetes dengan Machine Learning")
st.write("Masukkan informasi berikut untuk memprediksi risiko diabetes:")

# Input data dari pengguna
pregnancies = st.number_input('Masukkan Jumlah Kehamilan')
glucose = st.number_input('Masukkan Nilai Glukosa')
blood_pressure = st.number_input('Masukkan Nilai Tekanan Darah')
skin_thickness = st.number_input('Masukkan Ketebalan Kulit')
insulin = st.number_input('Masukkan Nilai Insulin')
bmi = st.number_input('Masukkan Indeks Massa Tubuh (BMI)')
dpf = st.number_input('Masukkan Diabetes Pedigree Function')
age = st.number_input('Masukkan Usia')

# Prediksi
if st.button("Prediksi Diabetes"):
    # Format input ke dalam array untuk model
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = diabetes_model.predict(input_data)
    if prediction[0] == 1:
        st.error("Hasil: Risiko Diabetes")
    else:
        st.success("Hasil: Tidak Berisiko Diabetes")
