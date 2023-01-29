import streamlit as st
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

#Read csv file 
data = pd.read_csv(r"C:\Users\Abdullah\Desktop\zip\fivver\order_3\app\DataET0.csv" , sep=';' , decimal=',')

#Fill null values
data.fillna(data.median(numeric_only=True).round(1), inplace=True)

#Creat year column
temp = pd.to_datetime(data["Date/heure"])
data['year'] = pd.DatetimeIndex(temp).year

st.subheader('Visualizing data features over the time period')



st.subheader('Average moy temprature over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['moy_Temp[°C]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average max temprature over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['max_Temp[°C]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average min temprature over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['min_Temp[°C]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average moy_DewPoint over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['moy_DewPoint[°C]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average min_DewPoint over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['min_DewPoint[°C]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average moy_SolarRadiation over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['moy_SolarRadiation[W/m2]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average moy_VPD over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['moy_VPD[kPa]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average min_VPD over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['min_VPD[kPa]'].mean())
st.line_chart(data=temp, x='year')



st.subheader('Average moy_RelativeHumidity over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['moy_RelativeHumidity[%]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average max_RelativeHumidity over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['max_RelativeHumidity[%]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average min_RelativeHumidity over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['min_RelativeHumidity[%]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average Somme_Precipitation over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['Somme_Precipitation[mm]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average moy_WindSpeed over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['moy_WindSpeed[m/s]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average max_WindSpeed over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['max_WindSpeed[m/s]'].mean())
st.line_chart(data=temp, x='year')


st.subheader('Average max_WindSpeedMax over the years')
temp = pd.DataFrame(data.groupby(data.year,as_index=False)['max_WindSpeedMax[m/s]'].mean())
st.line_chart(data=temp, x='year')


