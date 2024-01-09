# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing  # For preprocessing tasks
import matplotlib.pyplot as pltstrea
import seaborn as sns
import plotly.express as px
import streamlit as st
import io

def load_data(uploaded_file):
    """Reads the uploaded dataset and returns a Pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)  # Adjust for other file types if needed
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    
def handle_missing_values(df, method="fill_mean"):
    if(method=="drop_tuples"):
        df = df.dropna()
        pass

    if(method=="fill_with_mean"):
        try:
            df.fillna(df.mean(), inplace=True)
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Cant perform operation without numeric value: {e}")
            return None
        pass

    if(method=="fill_with_zero"):
        df = df. fillna (0)
        st.dataframe(df.head())
        pass
        
    if(method=="interpolate"):
        df = df. interpolate ()
        st.dataframe(df.head())
        pass
    # Display data info
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    pass


# Define additional functions for other preprocessing tasks

def create_histogram(df, column):
    # Create a histogram visualization for the specified column
    pass

def create_scatter_plot(df, x_column, y_column):
    # Create a scatter plot visualization
    pass

# Define functions for other visualization techniques

# Set up the main layout
st.title("Datavize")

#-------------------------------------
#-------------------------------------
#---------------UI--------------------
#-------------------------------------
#-------------------------------------
#-------------------------------------

# Uploading data
uploaded_file = st.file_uploader("Choose a dataset to upload")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    # Display data preview
    st.dataframe(df.head())
    # Display data info
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


#---------------------------selecting operations------------------
#--------------------------Preprocessing options------------------
    preprocess_options = st.multiselect("Select preprocessing operations:", ["Handle missing values", "Outlier detection", "Data Transformation"])
#-------------------------------------------
#--------Handiling missing values-----------
#-------------------------------------------
    if "Handle missing values" in preprocess_options:
        # Handle missing values based on user input
        preprocess_options = st.multiselect("Select method to remove missing values:", ["Drop Tuples", "Fill with mean", "Fill with Zero", "Interpolate Null Values"])
        if "Drop Tuples" in preprocess_options:
             handle_missing_values(df, method="drop_tuples")
             pass
        if "Fill with mean" in preprocess_options:
             handle_missing_values(df, method="fill_with_mean")
             pass
        if "Fill with Zero" in preprocess_options:
             handle_missing_values(df, method="fill_with_zero")
             pass
        if "Interpolate Null Values" in preprocess_options:
             handle_missing_values(df, method="interpolate")
             pass
        
#-------------------------------------------
#------------OutLier Detection--------------
#-------------------------------------------
    if "Outlier detection" in preprocess_options:
        try:
            Values = st.text_input("Enter A clomun")
            # Calculate IQR (Interquartile Range)
            Q1 = df[Values].quantile(0.25)
            Q3 = df[Values].quantile(0.75)
            IQR = Q3 - Q1
            # Define the lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Detect outliers
            outliers = df[(df[Values] < lower_bound) | (df[Values] > upper_bound)]
            st.write(outliers)
        except Exception as e:
            st.error(f"Enter Column Name: {e}")
        pass

#-------------------------------------------
#------------Normalization------------------
#-------------------------------------------
    if "Data Transformation" in preprocess_options:
        # Handle noisy data based on user input
        preprocess_options = st.multiselect("Select method to remove missing values:", ["Normalization with mix maxscaling", "One Hot Encoding", "Fill with Zero", "Interpolate Null Values"])
        if "Normalization with mix maxscaling" in preprocess_options:
             for column in df.columns:
                  df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                  st.dataframe(df.head())
             pass
        if "One Hot Encoding" in preprocess_options:
             # define one hot encoding
             Color = st.text_input("Enter A clomun that contains categoriucal values")
             one_hot_encoded = pd.get_dummies(df['Color'], prefix='Color')
             # Concatenate the one-hot encoded columns with the original DataFrame
             df = pd.concat([df, one_hot_encoded], axis=1)
             st.dataframe(df.head())
             pass
        if "Fill with Zero" in preprocess_options:
             handle_missing_values(df, method="fill_with_zero")
             pass
        if "Interpolate Null Values" in preprocess_options:
             handle_missing_values(df, method="interpolate")
             pass
        pass
# ... implement other preprocessing options


#-------------------------------------------------------------
#----------------------Visualization options------------------
#-------------------------------------------------------------
    visualization_options = st.multiselect("Select visualizations:", ["Histogram", "Scatter plot", ...])
    if "Histogram" in visualization_options:
        column = st.selectbox("Choose a column for the histogram:", df.columns)
        create_histogram(df, column)
    # ... implement other visualization options