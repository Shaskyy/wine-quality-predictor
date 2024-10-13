import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st

# Membaca dan mempersiapkan data
dfRedWine = pd.read_csv('winequality-red.csv')
x = dfRedWine.drop('quality', axis=1)
y = dfRedWine['quality']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Melatih model jika belum disimpan
import os
if not os.path.exists('model_redwine.pkl'):
    model_RedWine = LinearRegression()
    model_RedWine.fit(x_train, y_train)
    joblib.dump(model_RedWine, 'model_redwine.pkl')
else:
    model_RedWine = joblib.load('model_redwine.pkl')

# Judul aplikasi
st.title("Prediksi Kualitas Anggur Merah")

# Membuat input untuk setiap fitur
feature_names = x.columns.tolist()
input_data = []

for feature in feature_names:
    value = st.number_input(feature.capitalize(), format="%.2f")  # Input numerik
    input_data.append(value)

# Tombol untuk memprediksi
if st.button("Prediksi Kualitas"):
    input_df = pd.DataFrame([input_data], columns=x.columns)
    prediction = model_RedWine.predict(input_df)[0]
    st.success(f"Kualitas Prediksi: {prediction:.2f}")
