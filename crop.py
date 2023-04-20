### Importing Libraries ###

import pandas as pd
import seaborn as sns 
import numpy as np 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import pickle
import streamlit as st
import pickle

model=pickle.load(open('xgb.pkl', 'rb'))

st.title("Crop Prediction App")
Nitrogen =st.sidebar.slider("Nitrogen",1,100,1)
Phosporous=st.sidebar.slider("Phosporous", 1.0, 100.0, step=1.0) 
Potassium=st.sidebar.slider("Potassium",1.0, 100.0, step=1.0)
temperature=st.sidebar.slider("temperature", 1.0, 100.0, step=1.0) 
humidity=st.sidebar.slider("humidity", 1.0, 100.0, step=1.0) 
ph=st.sidebar.slider("ph", 0.0, 15.0, step=1.0)
rainfall=st.sidebar.slider("rainfall",100.0, 300.0,step=0.1)

df2=pd.DataFrame(data=[[Nitrogen,Phosporous,Potassium,temperature,humidity,ph,rainfall]],columns=['Nitrogen', 'Phosporous', 'Potassium', 'temperature', 'humidity', 'ph', 'rainfall'])


if st.button("Predict"):
    prediction=model.predict(df2)
    if prediction==0 :
        st.success("Cauliflower")
    elif prediction==1 :
        st.success("Onion")
    elif prediction==2 :
        st.success("Raddish")
    elif prediction==3 :
        st.success("Tomato")
    elif prediction==4 :
        st.success("Apple")  
    elif prediction==5 :
        st.success("Banana")
    elif prediction==6 :
        st.success("Blackgram")
    elif prediction==7 :
        st.success("Chickpea")
    elif prediction==8 :
        st.success("Coffee")
    elif prediction==22 :
        st.success("Rice")
    else:
        st.error("Watermelon")