import joblib
import numpy as np
import pandas as pd

encoder_Application_mode = joblib.load("model/encoder_Application_mode.joblib")
encoder_Course = joblib.load("model/encoder_Course.joblib")
encoder_Marital_status = joblib.load("model/encoder_Marital_status.joblib")
encoder_Nacionality = joblib.load("model/encoder_Nacionality.joblib")
encoder_Previous_qualification = joblib.load("model/encoder_Previous_qualification.joblib")
encoder_Fathers_occupation = joblib.load("model/encoder_Fathers_occupation.joblib")
encoder_Fathers_qualification = joblib.load("model/encoder_Fathers_qualification.joblib")
encoder_Mothers_occupation = joblib.load("model/encoder_Mothers_occupation.joblib")
encoder_Mothers_qualification = joblib.load("model/encoder_Mothers_qualification.joblib")
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

    df["Marital_status"] = encoder_Marital_status.transform(data["Marital_status"])
    df["Application_mode"] = encoder_Application_mode.transform(data["Application_mode"])
    df["Course"] = encoder_Course.transform(data["Course"])
    df["Previous_qualification"] = encoder_Previous_qualification.transform(data["Previous_qualification"])
    df["Nacionality"] = encoder_Nacionality.transform(data["Nacionality"])
    df["Mothers_qualification"] = encoder_Mothers_qualification.transform(data["Mothers_qualification"])
    df["Fathers_qualification"] = encoder_Fathers_qualification.transform(data["Fathers_qualification"])
    df["Mothers_occupation"] = encoder_Mothers_occupation.transform(data["Mothers_occupation"])
    df["Fathers_occupation"] = encoder_Fathers_occupation.transform(data["Fathers_occupation"])
    df["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1,1))[0]
    df["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1,1))[0]
    
    return df
