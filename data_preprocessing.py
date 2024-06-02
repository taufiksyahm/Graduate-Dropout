import joblib
import numpy as np
import pandas as pd

encoder_Application_mode = joblib.load("model/encoder_Application_mode.joblib")
encoder_Course = joblib.load("model/encoder_Course.joblib")
encoder_Fathers_occupation = joblib.load("model/encoder_Fathers_occupation.joblib")
encoder_Fathers_qualification = joblib.load("model/encoder_Fathers_qualification.joblib")
encoder_Marital_status = joblib.load("model/encoder_Marital_status.joblib")
encoder_Mothers_occupation = joblib.load("model/encoder_Mothers_occupation.joblib")
encoder_Mothers_qualification = joblib.load("model/encoder_Mothers_qualification.joblib")
encoder_Nacionality = joblib.load("model/encoder_Nacionality.joblib")
encoder_Previous_qualification = joblib.load("model/encoder_Previous_qualification.joblib")
encoder_target = joblib.load("model/encoder_target.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler_Curricular_units_2nd_sem_grade.joblib")
    
def data_preprocessing(data):
    """Preprocessing data
     
    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
            
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    
    data = data.copy()
    df = pd.DataFrame()
    
    df["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1,1))[0]
    df["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1,1))[0]
        
    df["Application_mode"] = encoder_Application_mode.transform(data["Application_mode"])[0]
    df["Course"] = encoder_Course.transform(data["Course"])[0]
    df["Fathers_occupation"] = encoder_Fathers_occupation.transform(data["Fathers_occupation"])[0]
    df["Fathers_qualification"] = encoder_Fathers_qualification.transform(data["Fathers_qualification"])[0]
    df["Marital_status"] = encoder_Marital_status.transform(data["Marital_status"])[0]
    df["Mothers_occupation"] = encoder_Mothers_occupation.transform(data["Mothers_occupation"])[0]
    df["Mothers_qualification"] = encoder_Mothers_qualification.transform(data["Mothers_qualification"])[0]
    df["Nacionality"] = encoder_Nacionality.transform(data["Nacionality"])[0]
    df["Previous_qualification"] = encoder_Previous_qualification.transform(data["Previous_qualification"])[0]
        
    return df
