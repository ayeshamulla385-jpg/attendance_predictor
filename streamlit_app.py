import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("Attendance Predictor System")

study = st.slider("Study Hours",1,10)
sleep = st.slider("Sleep Hours",4,10)
distance = st.slider("Distance",1,30)
attendance = st.slider("Previous Attendance",40,100)

participation = st.selectbox(
    "Participation",
    [0,1]
)

health = st.selectbox(
    "Health Issue",
    [0,1]
)

if st.button("Predict"):

    data = np.array([[
        study,
        sleep,
        distance,
        attendance,
        participation,
        health
    ]])

    prediction = model.predict(data)

    result = encoder.inverse_transform(prediction)

    st.success(f"Attendance: {result[0]}")