# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing  # For preprocessing tasks
import matplotlib.pyplot as pltstrea
import seaborn as sns
import plotly.express as px
import streamlit as st
import io
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
import matplotlib.pyplot as plt

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

def create_distplot(df, column):
    # Create a Graph Plot visualization for the specified column
    plot=sns.distplot(df[column], bins=10, kde=True, rug=False)
    st.pyplot(plot.get_figure())
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
            Values = st.selectbox("Choose a column", df.columns)
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
#------------Data transformation------------
#-------------------------------------------
    if "Data Transformation" in preprocess_options:
        # Handle noisy data based on user input
        preprocess_options = st.multiselect("Select method to remove missing values:", ["Normalization with mix maxscaling", "One Hot Encoding", "Ordinal Encoding", "Nominal Encoding"])
        if "Normalization with mix maxscaling" in preprocess_options:
             for column in df.columns:
                  df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                  st.dataframe(df.head())
             pass
        if "One Hot Encoding" in preprocess_options:
             st.dataframe(df.head())
             try:
                 # define one hot encoding
                 column = st.selectbox("Choose a column for the Graph Plot:", df.columns)
                 one_hot_encoded = pd.get_dummies(df[column], prefix=column)
                 # Concatenate the one-hot encoded columns with the original DataFrame
                 df = pd.concat([df, one_hot_encoded], axis=1)
                 st.dataframe(df.head())
             except Exception as e:
                 st.error(f"Error loading data: {e}")
             pass
        if "Ordinal Encoding" in preprocess_options:
             st.dataframe(df.head())
             try:
                 # define one hot encoding
                 column = st.selectbox("Choose a column for the Graph Plot:", df.columns)
                 # Create an ordinal encoder instance
                 ordinal_encoder = ce.OrdinalEncoder(cols=[column])
                 # Apply ordinal encoding on the 'Size' column
                 df[column+'_encoded'] = ordinal_encoder.fit_transform(df[column])

                 st.dataframe(df.head())
             except Exception as e:
                 st.error(f"Error loading data: {e}")
             pass
        if "Nominal Encoding" in preprocess_options:
             st.dataframe(df.head())
             try:
                 column = st.selectbox("Choose a column for the Graph Plot:", df.columns)
                 # Perform one-hot encoding (Nominal encoding) on the 'Color' column
                 nominal_encoded = pd.get_dummies(df[column], prefix=column)
                 # Concatenate the nominal encoded columns with the original DataFrame
                 df = pd.concat([df, nominal_encoded], axis=1)
                 st.dataframe(df.head())
             except Exception as e:
                 st.error(f"Error loading data: {e}")
             pass
        pass
# ... implement other preprocessing options


#-------------------------------------------------------------
#----------------------Visualization options------------------
#-------------------------------------------------------------
    visualization_options = st.multiselect("Select visualizations:", ["Graph Plot on full DataSet","Graph Plot", "Box Plot", "Violin Plot", "Line Plot", ...])
    if "Graph Plot" in visualization_options:
        column = st.selectbox("Choose a column for the Graph Plot:", df.columns)
        create_distplot(df, column)
        pass

    if "Graph Plot on full DataSet" in visualization_options:
        # Create a Graph Plot visualization for the specified column
        plot=sns.distplot(df, bins=10, kde=True, rug=False)
        st.pyplot(plot.get_figure())
        pass

    if "Box Plot" in visualization_options:
        column = st.selectbox("Choose a column for the Box Plot:", df.columns)
        # Create a Graph Plot visualization for the specified column
        plt.boxplot(df[column])
        plt.xlabel(column)
        plt.ylabel('Value')
        plt.title('Boxplot of '+column)
        st.pyplot(plt.gcf())
        pass

    if "Violin Plot" in visualization_options:
        column = st.selectbox("Choose a column for the Violin Plot:", df.columns)
        try:
            selected_attribute_values = df[column]
            # Create a violin plot
            sns.violinplot(x=selected_attribute_values)  # Create the violin plot
            plt.title('Violin Plot of '+column)  # Set the title
            plt.xlabel(column)  # Set the x-axis label
            plt.ylabel('Values')  # Set the y-axis label
            st.pyplot(plt.gcf())
        except Exception as e:
                 st.error(f"Error loading data: {e}")
        pass

    if "Line Plot" in visualization_options:
        column = st.selectbox("Choose a column for the Line Plot:", df.columns)
        try:
            attribute_values = df[column]
            # Plotting a line plot
            plt.plot(attribute_values, marker='o', linestyle='-', color='b')  # Plotting the line
            plt.title('Line Plot of '+column)  # Set the title
            plt.xlabel('Index')  # Set the x-axis label
            plt.ylabel(column)  # Set the y-axis label
            plt.grid(True)  # Show grid
            st.pyplot(plt.gcf())
        except Exception as e:
                 st.error(f"Error loading data: {e}")
        pass

    # ... implement other visualization options