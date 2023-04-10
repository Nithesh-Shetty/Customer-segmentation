# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:16:08 2023

@author: Nithesh Shetty
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:25:48 2023

@author: Nithesh Shetty
"""
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler



arr = np.array([[7.86270620e+04, 1.36199812e+03, 7.59398496e-01, 3.62781955e-01,
        6.37218045e-01, 1.39097744e-01, 4.73684211e-01, 3.87218045e-01],
       [4.08297484e+04, 2.02891234e+02, 1.29870130e-01, 3.71753247e-01,
        6.28246753e-01, 1.16883117e-01, 6.21753247e-01, 2.61363636e-01],
       [5.92731503e+04, 7.25705047e+02, 2.27129338e-01, 3.35962145e-01,
        6.64037855e-01, 5.52050473e-02, 5.31545741e-01, 4.13249211e-01],
       [2.33850573e+04, 7.28233945e+01, 7.33944954e-02, 3.46330275e-01,
        6.53669725e-01, 2.66055046e-01, 5.87155963e-01, 1.46788991e-01]])

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

#Standardization of the data
scaler = StandardScaler()
scaled_df1 = scaler.fit_transform(df1)


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