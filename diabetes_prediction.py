# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:21:27 2024
@author: nihqd
"""
import numpy as np
import pickle 
import streamlit as st
import requests
from io import BytesIO

# Download the model from GitHub
model_url = 'https://github.com/Niihaad/ML_Algorithms/raw/main/trained_modelIntell.sav'
response = requests.get(model_url)

# Load the model from the downloaded content
if response.status_code == 200:
    loaded_model = pickle.load(BytesIO(response.content))

# Function for prediction
def smartness_prediction(input_data):
    predicted_outcome = loaded_model.predict(input_data)
    return f'Predicted smartness level: {predicted_outcome[0]}'

def main():
    st.title('Test Your Level of Smartness By Nihad')
    
    usingGpt = st.text_input('On a scale from 0 to 1, how frequently do you use ChatGPT?')
    usingGoogle = st.text_input('On a scale from 0 to 1, how often do you rely on Google?')
    usingMind = st.text_input('On a scale from 0 to 1, how frequently do you engage your mind?')
    
    total = ''
    if st.button("Predict Smartness"):
        try:
            # Convert inputs to floats
            gpt_val = float(usingGpt)
            google_val = float(usingGoogle)
            mind_val = float(usingMind)
            input_data = np.array([gpt_val, google_val, mind_val]).reshape(1, -1)

            # Perform prediction
            total = smartness_prediction(input_data)
        except ValueError as e:
            st.error(f"Error: {e}. Please enter valid numerical values.")
            total = "Please enter valid numerical values."

    st.success(total)
    
if __name__=='__main__':
    main()

    
