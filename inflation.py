import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Pakistan Inflation Rate Prediction", page_icon="🌍", layout="wide")

    # Page title and description
    st.title("Pakistan Inflation Rate Prediction 📊")
    st.markdown("""This application analyzes historical inflation rates in Pakistan and predicts future rates using machine learning. 
    Upload your dataset and explore the insights!""")

    # File upload
    uploaded_file = st.file_uploader("Upload the inflation dataset (CSV format)", type="csv")
    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Clean and preprocess the data
        data['Inflation Rate (%)'] = data['Inflation Rate (%)'].str.replace('%', '').astype(float)
        data['Year'] = data['Year'].astype(int)

        # Display the dataset
        st.subheader("Dataset Preview")
        st.dataframe(data.style.format({"Inflation Rate (%)": "{:.2f}"}))

        # Prepare features and target
        X = data[['Year']]
        y = data['Inflation Rate (%)']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("Model Evaluation")
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")

        # Predict future inflation rates
        future_years = pd.DataFrame({'Year': range(2025, 2031)})
        future_predictions = model.predict(future_years)

        # Display predictions
        predictions_df = pd.DataFrame({
            'Year': future_years['Year'],
            'Predicted Inflation Rate (%)': future_predictions
        })
        st.subheader("Future Predictions")
        st.table(predictions_df.style.format({"Predicted Inflation Rate (%)": "{:.2f}"}))

        # Visualization
        st.subheader("Visualization")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Historical Data")
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=data['Year'], y=data['Inflation Rate (%)'], marker="o", label="Actual Data", color="blue")
            plt.xlabel("Year")
            plt.ylabel("Inflation Rate (%)")
            plt.title("Historical Inflation Rates")
            plt.legend()
            st.pyplot(plt)

        with col2:
            st.markdown("### Future Predictions")
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=predictions_df['Year'], y=predictions_df['Predicted Inflation Rate (%)'], marker="o", color="green", label="Predictions")
            plt.xlabel("Year")
            plt.ylabel("Predicted Inflation Rate (%)")
            plt.title("Predicted Inflation Rates")
            plt.legend()
            st.pyplot(plt)

        # Combined Visualization
        st.markdown("### Combined Historical and Future Predictions")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=data['Year'], y=data['Inflation Rate (%)'], marker="o", label="Historical Data", color="blue")
        sns.lineplot(x=predictions_df['Year'], y=predictions_df['Predicted Inflation Rate (%)'], marker="o", label="Future Predictions", color="green")
        plt.axvline(x=2024, color='red', linestyle='--', label="2024 (Prediction Start)")
        plt.xlabel("Year")
        plt.ylabel("Inflation Rate (%)")
        plt.title("Historical and Predicted Inflation Rates")
        plt.legend()
        st.pyplot(plt)

        # Download button for predictions
        st.markdown("### Download Predictions")
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Predictions as CSV", data=csv, file_name="predicted_inflation_rates.csv", mime="text/csv")

    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
