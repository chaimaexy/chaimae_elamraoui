# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:21:27 2024
@author: nihqd
"""
import numpy as np
import pickle 
import streamlit as st

loaded_model = pickle.load(open('D:\\FirstDeployment\\trained_model.sav', 'rb'))

#creating a function for prediction
def diabetics_prediction(input_data):
   
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return('The person is not diabetic')
    else:
     return('The person is diabetic')
 

def main():
    #setting a ti = st.text_input('Set the number of Preganancies')tle for our app
    st.title('Diabetes Prediction By Nihad')
    #getting the input from user
    
    Pregnancies = st.text_input('Set the number of Preganancies')
    Glucose = st.text_input('Set the level of Glucose')
    BloodPressure = st.text_input('Set the  value of BloodPressure')
    
    SkinThickness = st.text_input('Set the value of SkinThickness')
    Insulin = st.text_input('Set the level of Insulin')
    BMI = st.text_input('Set the BMI')
    DiabetesPedigreeFunction  = st.text_input('Set the degree of DiabetesPedigreeFunction')
    Age  = st.text_input('Set your age')
    
    
    
    #prediction code
    diagnosis = ''
    
    #Button of prediction
    if st.button("Diabetes Test"):
        diagnosis = diabetics_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    #only when it is running from standalone file 
if  __name__=='__main__':
    main()
    
    