import streamlit as st
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

#Read csv file 
# st.subheader('Upload your data file')
# uploaded_file = st.file_uploader("")
# if uploaded_file is not None:
#   data = pd.read_csv(uploaded_file , sep=';' ,decimal=',')
data = pd.read_csv(r"C:\Users\Abdullah\Desktop\zip\fivver\order_3\app\DataET0.csv" , sep=';' , decimal=',')

#display raw data 
st.subheader('Raw data')
st.write(data.head())

#Display the number of missing values for each column
st.subheader('Number of missing values for each column')
null_vals = pd.DataFrame(data.isna().sum() , columns = ['No. of null values'])
st.write(null_vals)

#Visualize the number of missing values for each column
st.subheader('Visualize number of missing values for each column')
st.bar_chart(null_vals)


#Dropping unnecessary columns 
data.drop('Date/heure', inplace=True, axis=1)
data.drop('moy_WindDirection[deg]', inplace=True, axis=1)
data.drop('dernier_WindDirection[deg]', inplace=True, axis=1)


#Filiing null values with the median value
data.fillna(data.median(numeric_only=True).round(1), inplace=True)

#Display dataset after preprocessing 
st.subheader('Data after preprocessing(drop unnecessary columns, fill nan vals, etc)')
st.write(data.head())

#Display datatype of each column of the dataset
st.subheader('Datatype of each column of the dataset')
data_types = pd.DataFrame(data.dtypes , columns = ['Data_types'])
st.write(data_types)