
# LIBRARIES:
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

import base64
import io


# PAGE LAYOUT AND TITLE:
st.set_page_config(page_title='Airline Prediction. Best Machine Learning Model',
    layout='wide')


# MODEL BUILDING:
def model_building(data):

    # Define X and y
    X = data.iloc[:,:-1] # Selects all cols except the last one
    y = data.iloc[:,-1] # Selects only last col of the df

    # Display a head of the data
    st.markdown('**Data head:**')
    st.info(data.head())

    # Display shape of X and y
    st.markdown('**Data shape:**')
    st.info(X.shape)
    st.info(y.shape)

    # Display cols for X and y
    st.markdown('**X and y varibles:**')
    st.info(list(X.columns()))
    st.info(list(y.columns()))