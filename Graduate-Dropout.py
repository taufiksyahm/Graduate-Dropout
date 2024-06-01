import streamlit as st
import pandas as pd
import joblib
#from data_preprocessing import data_preprocessing, encoder_Application_mode, encoder_Course, encoder_Fathers_occupation, encoder_Fathers_qualification, encoder_Marital_status, encoder_Mothers_occupation, encoder_Mothers_qualification, encoder_Nacionality, encoder_Previous_qualification
#from prediction import prediction

col1 = st.columns([1, 5])
with col1:
    st.header('Graduate-Dropout App (Prototype)')

    data = pd.DataFrame()
     
col1, col2, col3 = st.columns(3)
     
with col1:
    Application_mode = st.selectbox(label='Application_mode', options=encoder_Application_mode.classes_, index=1)
    data["Application_mode"] = [Application_mode]
     
with col2:
    Course = st.selectbox(label='Course', options=encoder_Course.classes_, index=1)
    data["Course"] = [Course]
     
with col3:
    Marital_status = st.selectbox(label='Marital_status', options=encoder_Marital_status.classes_, index=5)
    data["Marital_status"] = Marital_status

col1, col2 = st.columns(2)
     
with col1:
    Nacionality = st.selectbox(label='Nationality', options=encoder_Nacionality.classes_, index=1)
    data["Nacionality"] = [Nacionality]
     
with col2:
    Previous_qualification = st.selectbox(label='Previous_qualification', options=encoder_Previous_qualification.classes_, index=1)
    data["Previous_qualification"] = [Previous_qualification]
   
col1, col2 = st.columns(2)
     
with col1:
    Curricular_units_1st_sem_approved = int(st.number_input(label='Curricular_units_1st_sem_approved', value=0))
    data["Curricular_units_1st_sem_approved"] = Curricular_units_1st_sem_approved
     
with col2:
    Curricular_units_1st_sem_grade = int(st.number_input(label='Curricular_units_1st_sem_grade', value=0))
    data["Curricular_units_1st_sem_grade"] = Curricular_units_1st_sem_grade

col1, col2 = st.columns(2)
     
with col1:
    Curricular_units_2nd_sem_approved = int(st.number_input(label='Curricular_units_2nd_sem_approved', value=0))
    data["Curricular_units_2nd_sem_approved"] = Curricular_units_2nd_sem_approved
     
with col2:
    Curricular_units_2nd_sem_grade = int(st.number_input(label='Curricular_units_2nd_sem_grade', value=0))
    data["Curricular_units_2nd_sem_grade"] = Curricular_units_2nd_sem_grade
     
with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Status Prediction: {}".format(prediction(new_data)))
