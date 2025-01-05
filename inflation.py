import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def main():
    st.title("Global Inflation Rate Prediction")

    st.markdown("""
    **Welcome to the Global Inflation Rate Prediction tool!**
    This app predicts future inflation rates for a selected country using historical data.
    Please upload the dataset in CSV format to get started.
    """)

    # User information input
    st.sidebar.subheader("User Information")
    user_name = st.sidebar.text_input("Enter your name:")
    user_age = st.sidebar.number_input("Enter your age:", min_value=1, step=1)
    user_email = st.sidebar.text_input("Enter your email:")

    if user_name and user_email:
        st.sidebar.success(f"Welcome, {user_name}!")

    # File upload
    uploaded_file = st.file_uploader("Upload your inflation dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Replace blank spaces and NaN values with 0
        data = data.replace(r'^\s*$', 0, regex=True).fillna(0)

        # Dataset overview with vertical slider
        st.subheader("Dataset Overview")
        total_rows = data.shape[0]
        num_rows = st.slider(
            "Select the number of rows to display:",
            min_value=5,
            max_value=total_rows,
            value=5,
            step=1
        )
        st.write(data.head(num_rows))  # Display the selected number of rows

        # Ensure required columns exist
        if 'country_name' in data.columns and 'indicator_name' in data.columns:
            # Select a country
            country = st.selectbox("Select a country for prediction:", data['country_name'].unique())

            # Filter data for the selected country
            country_data = data[data['country_name'] == country].drop('indicator_name', axis=1)

            # Reshape the data for modeling
            try:
                country_data = country_data.set_index('country_name').transpose().reset_index()
                country_data.co
