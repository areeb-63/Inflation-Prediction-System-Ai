import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

    # File upload
    uploaded_file = st.file_uploader("Upload your inflation dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Display the dataset
        st.subheader("Dataset Overview")
        st.write(data.head())

        # Ensure required columns exist
        if 'country_name' in data.columns and 'indicator_name' in data.columns:
            # Select a country
            country = st.selectbox("Select a country for prediction:", data['country_name'].unique())

            # Filter data for the selected country
            country_data = data[data['country_name'] == country].drop('indicator_name', axis=1)

            # Reshape the data for modeling
            country_data = country_data.set_index('country_name').transpose().reset_index()
            country_data.columns = ['Year', 'Inflation Rate (%)']
            country_data['Year'] = country_data['Year'].astype(int)
            country_data['Inflation Rate (%)'] = country_data['Inflation Rate (%)'].astype(float)

            st.subheader(f"Historical Data for {country}")
            st.write(country_data)

            # Prepare features and target
            X = country_data[['Year']]
            y = country_data['Inflation Rate (%)']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            st.subheader("Model Evaluation")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")

            # User input for selecting prediction years
            st.markdown("### Predict Future Inflation Rates")
            start_year = st.slider("Select starting year for prediction:", min_value=2023, max_value=2040, value=2023)
            end_year = st.slider("Select ending year for prediction:", min_value=2024, max_value=2041, value=2024)

            # Create future years for prediction
            future_years = pd.DataFrame({'Year': range(start_year, end_year + 1)})
            future_predictions = model.predict(future_years)

            # Display predictions
            predictions_df = pd.DataFrame({
                'Year': future_years['Year'],
                'Predicted Inflation Rate (%)': future_predictions
            })
            st.subheader("Future Inflation Predictions")
            st.write(predictions_df)

            # Visualization
            st.subheader("Visualization")
            plt.figure(figsize=(10, 6))
            plt.plot(country_data['Year'], country_data['Inflation Rate (%)'], label='Historical Data', marker='o')
            plt.plot(future_years['Year'], future_predictions, label='Predicted Data', linestyle='--')
            plt.xlabel('Year')
            plt.ylabel('Inflation Rate (%)')
            plt.title(f'Inflation Rate Prediction for {country}')
            plt.legend()
            st.pyplot(plt)

            # Option to download predictions as a CSV
            st.markdown("### Download Predictions")
            st.download_button(
                label="Download Prediction Data",
                data=predictions_df.to_csv(index=False),
                file_name=f"{country}_inflation_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("The uploaded file does not contain the required columns: 'country_name' and 'indicator_name'.")
    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
