import streamlit as st
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVR
import seaborn as sns

#Read data
data = pd.read_csv(r"C:\Users\Abdullah\Desktop\zip\fivver\order_3\app\DataET0.csv" , sep=';' , decimal=',')

#Dropping unnecessary columns 
data.drop('Date/heure', inplace=True, axis=1)
data.drop('moy_WindDirection[deg]', inplace=True, axis=1)
data.drop('dernier_WindDirection[deg]', inplace=True, axis=1)

#Filiing null values with the median value
data.fillna(data.median(numeric_only=True).round(1), inplace=True)

#Extracting the ground truth column as labels 
Lables = data["ETP quotidien [mm]"]
data.drop("ETP quotidien [mm]", inplace=True, axis=1)

#Splitting training testing data
X_train, X_test, Y_train, Y_test = train_test_split(data, Lables, test_size=0.2, random_state=42)

#Encoding the labels for training and testing
label_encoder = preprocessing.LabelEncoder()
Y_Train_transformed = label_encoder.fit_transform(Y_train)
Y_Test_transformed = label_encoder.fit_transform(Y_test)

# Create SVR regression object
SVR_regressor = SVR(C=1.0, epsilon=0.2)

# Train SVR regression
SVR_regressor.fit(X_train,Y_Train_transformed)

#Predict the response for test dataset
SVR_Regressor_preds = SVR_regressor.predict(X_test)

st.title('ETP Quotidien Predictor')

st.subheader('Enter the following values')


moy_Temp = st.number_input('Enter moy_Temp')

max_Temp = st.number_input('Enter max_Temp')

min_Temp = st.number_input('Enter min_Temp')

moy_DewPoint = st.number_input('Enter moy_DewPoint')

min_DewPoint = st.number_input('Enter min_DewPoint')

moy_SolarRadiation = st.number_input('Enter moy_SolarRadiation')

moy_VPD = st.number_input('Enter moy_VPD')

min_VPD = st.number_input('Enter min_VPD')

moy_RelativeHumidity = st.number_input('Enter moy_RelativeHumidity')

max_RelativeHumidity = st.number_input('Enter max_RelativeHumidity')

min_RelativeHumidity = st.number_input('Enter min_RelativeHumidity')

Somme_Precipitation = st.number_input('Enter Somme_Precipitation')

moy_WindSpeed = st.number_input('Enter moy_WindSpeed')

max_WindSpeed = st.number_input('Enter max_WindSpeed')

max_WindSpeedMax = st.number_input('Enter max_WindSpeedMax')

if st.button('Get Predictions'):

    pred_list=[int(moy_Temp),
    int(max_Temp),
    int(min_Temp),
    int(moy_DewPoint),
    int(min_DewPoint),
    int(moy_VPD),
    int(min_VPD),
    int(moy_SolarRadiation),
    int(moy_RelativeHumidity),
    int(max_RelativeHumidity),
    int(min_RelativeHumidity),
    int(Somme_Precipitation),
    int(moy_WindSpeed),
    int(max_WindSpeed),
    int(max_WindSpeedMax)]
    
    vals = np.array(pred_list)
    
    vals = vals.reshape(1, -1)
    
    st.write("Input values are : " , pred_list)
    
    SVR_Regressor_preds = SVR_regressor.predict(vals)
    
    st.write("ETP Quotidien is : " , float(SVR_Regressor_preds))
