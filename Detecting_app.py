# Import necessary libraries.

import pickle
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import time as t
import streamlit as st


def app() :
    
    st.title("Credit Card Fraud Detection")
    st.image('hack-fraud-card-code.jpg')
    st.markdown("### Enter the details")
   
   # Create a form for user input.

    with st.form(key='form', clear_on_submit=True):

        category = st.number_input("category")
        amt = st.number_input("Amount")
        gender = st.selectbox("Gender", [0, 1])
        state = st.number_input("State")
        lat = st.number_input("Latitude")
        long = st.number_input("Longitude")
        city_pop = st.number_input("City Population")
        Hour = st.number_input("Hour")
        Day = st.number_input("Day")
        Age = st.slider("Age", 0, 100)

        lst = [category, amt, gender, state, lat, long, city_pop, Hour, Day, Age]

        submit = st.form_submit_button("Predict")

    if submit:                                                        # Handle the form submission.
        with open('Model.pkl', 'rb') as f:
            model = pickle.load(f)                                    # Load a machine learning model from 'Model.pkl'
        
        features = np.asarray(lst, dtype=np.float64)
        prediction = model.predict(features.reshape(1,-1))

        if prediction[0] == 0:                                        
            st.success("Legitimate Transaction")
            st.balloons()
        else:       
            st.warning("Fradulant Transaction")       


if __name__ == '__main__':
    app()
