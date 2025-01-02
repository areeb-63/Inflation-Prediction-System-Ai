import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Streamlit app
def main():
    st.title("Pakistan Inflation Rate Prediction")

    # File upload
    uploaded_file = st.file_uploader("Upload the inflation dataset (CSV format)", type="csv")
    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Clean and preprocess the data
        data['Inflation Rate (%)'] = data['Inflation Rate (%)'].str.replace('%', '').astype(float)
        data['Year'] = data['Year'].astype(int)

        # Display the dataset
        st.subheader("Dataset")
        st.write(data)

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
        st.write(f"Mean Squared Error: {mse}")

        # User input for selecting prediction years
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
        st.subheader("Future Predictions")
        st.write(predictions_df)

        # Visualization
        st.subheader("Choose the type of visualization:")
        plot_type = st.selectbox("Select Plot Type", ["Line Graph", "Bar Graph", "Heatmap"])

        # Create the plot based on user selection
        if plot_type == "Line Graph":
            # Line graph of actual vs predicted inflation rate
            plt.figure(figsize=(10, 6))
            plt.plot(data['Year'], data['Inflation Rate (%)'], color='blue', label='Actual Inflation Rate')
            plt.plot(data['Year'], model.predict(data[['Year']]), color='red', label='Trend Line')
            plt.xlabel('Year')
            plt.ylabel('Inflation Rate (%)')
            plt.title('Inflation Rate Prediction (Line Graph)')
            plt.legend()
            st.pyplot(plt)

        elif plot_type == "Bar Graph":
            # Bar graph of actual vs predicted inflation rate for future years
            plt.figure(figsize=(10, 6))
            plt.bar(data['Year'], data['Inflation Rate (%)'], color='blue', label='Actual Data')
            plt.bar(future_years['Year'], future_predictions, color='green', alpha=0.5, label='Predicted Data')
            plt.xlabel('Year')
            plt.ylabel('Inflation Rate (%)')
            plt.title('Inflation Rate Prediction (Bar Graph)')
            plt.legend()
            st.pyplot(plt)

        elif plot_type == "Heatmap":
            # Heatmap of the correlation matrix (useful if there are more features in the future)
            st.subheader("Correlation Heatmap")
            corr_matrix = data.corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title('Correlation Heatmap')
            st.pyplot(plt)

    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
