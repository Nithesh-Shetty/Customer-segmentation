# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:25:48 2023

@author: Nithesh Shetty
"""
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler


df=pd.read_excel("centroid_rev.xlsx")
arr = df.to_numpy()

st.set_page_config(page_title='My App', page_icon=':smiley:', layout='wide')
st.title('Customer segmentation')


st.sidebar.header('User Input Parameters')
def user_input_features():
    Income = st.sidebar.number_input("Enter Income in USD")
    Total_amt_spent=st.sidebar.number_input("Enter Total Amount Spent")
    AcceptedCmp=st.sidebar.number_input("Enter total campaigns accepted")
    Age = st.sidebar.number_input('Enter age')
    Marital_Status=st.sidebar.selectbox('Marital_Status',("Partner","Single"))
   
    data = {'Income':Income,
            'Total_amt_spent':Total_amt_spent,
            'AcceptedCmp':AcceptedCmp,
            'Age':Age,
            'Marital_Status': Marital_Status
            }
    features = pd.DataFrame(data,index = [0])
    return features 

df1 = user_input_features()
st.subheader('User Input parameters')
st.write(df1)


# Creating bins of different age group
bin_edges = [20,40,60,100]
bin_labels = ["20-40", '40-60', '60+']
df1['Age_group'] = pd.cut(df1['Age'], bins=bin_edges, labels=bin_labels)
df1.drop("Age",axis=1,inplace=True)

if df1["Marital_Status"][0]=="Single":
  df1["Marital_status_single"], df1["Marital_status_Partner"]=[1],[0]
else:
  df1["Marital_status_single"], df1["Marital_status_Partner"]=[0],[1]
df1.drop("Marital_Status",axis=1,inplace=True)
df1=pd.get_dummies(df1)
df1=df1.to_numpy()

# find the nearest centroid to the new point
distances = np.sqrt(((df1-arr)**2).sum(axis=1))
nearest_centroid_index = np.argmin(distances)

st.subheader('Predicted Result')
if nearest_centroid_index==0:
    st.write("Customer belongs to cluster 0")
elif nearest_centroid_index==1:
    st.write("Customer belongs to cluster 1")
elif nearest_centroid_index==2:
    st.write("Customer belongs to cluster 2")
else:
    st.write("Customer belongs to cluster 3")


    
    









