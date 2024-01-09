# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 23:32:32 2024

@author: nihqd
"""

import numpy as np
import pickle 
import streamlit as st

import requests
from io import BytesIO

# Replace 'Niihaad', 'ML_Algorithms', and 'trained_modelIntell.sav' with your details
github_username = 'Niihaad'
repo_name = 'ML_Algorithms'
file_path = 'trained_modelIntell.sav'

# Construct the raw file URL on GitHub
raw_url = f'https://raw.githubusercontent.com/{github_username}/{repo_name}/main/{file_path}'

# Fetch the model file from GitHub
response = requests.get(raw_url)

# Load the model from the response content
loaded_model = pickle.load(BytesIO(response.content))


def smartness_prediction(input_data):
    # Example of input_data = [[0.9, 0.7, 0.1]]
    predicted_outcome = loaded_model.predict(input_data)
    return 'You are ' + str(predicted_outcome[0])  # Convert the prediction to a string


def main():
    st.title('Test Your Level of Smartness By Nihad ')
    
    usingGpt = st.text_input('On a scale from 0 to 1, how frequently do you use ChatGPT ?')
    usingGoogle = st.text_input('On a scale from 0 to 1, how often do you rely on Google ?')
    usingMind = st.text_input('On a scale from 0 to 1, how frequently do you engage your mind ?')
    
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
            total = "Please enter valid numerical values ."

    st.success(total)
    
    #only when it is running from standalone file 
if  __name__=='__main__':
    main()
    
