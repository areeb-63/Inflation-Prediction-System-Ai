import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit app
def main():
    st.title("Pakistan Inflation Rate Prediction")

    # Load the dataset
    file_path = "Pakistan Inflation Rate 1960 to 2024.csv"
    data = pd.read_csv(file_path)

    # Clean and preprocess the data
    data['Inflation Rate (%)'] = data['Inflation Rate (%)'].str.replace('%', '').astype(float)
    data['Year'] = data['Year'].astype(int)

    # Display the dataset
    st.subheader("Dataset")
    st.write(data)

    # Prepare the features (X) and target (y)
    X = data[['Year']]
    y = data['Inflation Rate (%)']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error: {mse}")

    # Predict future inflation rates (e.g., 2025-2030)
    future_years = pd.DataFrame({'Year': range(2025, 2031)})
    future_predictions = model.predict(future_years)

    # Display future predictions
    predictions_df = pd.DataFrame({
        'Year': future_years['Year'],
        'Predicted Inflation Rate (%)': future_predictions
    })
    st.subheader("Future Predictions")
    st.write(predictions_df)

    # Plot the results
    st.subheader("Visualization")
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Year'], data['Inflation Rate (%)'], color='blue', label='Actual Data')
    plt.plot(data['Year'], model.predict(data[['Year']]), color='red', label='Trend Line')
    plt.scatter(future_years['Year'], future_predictions, color='green', label='Future Predictions')
    plt.xlabel('Year')
    plt.ylabel('Inflation Rate (%)')
    plt.title('Inflation Rate Prediction')
    plt.legend()
    st.pyplot(plt)

if __name__ == "__main__":
    main()
