import numpy as np
import pickle 
import streamlit as st
import requests
from io import BytesIO

def load_model_from_github(model_url):
    st.write("Fetching the model...")
    response = requests.get(model_url)
    if response.status_code == 200:
        try:
            loaded_model = pickle.load(BytesIO(response.content))
            st.write("Model loaded successfully!")
            return loaded_model
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            return None
    else:
        st.error("Failed to fetch the model")
        return None

def smartness_prediction(input_data, model):
    if model:
        predicted_outcome = model.predict(input_data)
        return f'Predicted smartness level: {predicted_outcome[0]}'
    else:
        return "Model not loaded properly"

def main():
    st.title('Test Your Level of Smartness By Nihad')
    model_url = 'https://github.com/Niihaad/ML_Algorithms/blob/main/trained_modelIntell.sav'
    loaded_model = load_model_from_github(model_url)
    
    usingGpt = st.text_input('On a scale from 0 to 1, how frequently do you use ChatGPT?')
    usingGoogle = st.text_input('On a scale from 0 to 1, how often do you rely on Google?')
    usingMind = st.text_input('On a scale from 0 to 1, how frequently do you engage your mind?')
    
    total = ''
    if st.button("Predict Smartness"):
        try:
            gpt_val = float(usingGpt)
            google_val = float(usingGoogle)
            mind_val = float(usingMind)
            input_data = np.array([gpt_val, google_val, mind_val]).reshape(1, -1)

            total = smartness_prediction(input_data, loaded_model)
        except ValueError as e:
            st.error(f"Error: {e}. Please enter valid numerical values.")
            total = "Please enter valid numerical values."

    st.success(total)

if __name__=='__main__':
    main()
