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

    # Built-in dataset (replace this with actual dataset)
    data = pd.DataFrame({
        'Year': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        'Inflation Rate (%)': [4.5, 3.9, 4.0, 4.2, 5.0, 6.5, 7.0, 9.0, 12.0, 6.0, 10.0, 12.5, 7.5, 8.0, 9.0, 4.5, 3.8, 4.6, 5.4, 9.4, 8.0]
    })

    # Clean and preprocess the data
    data['Inflation Rate (%)'] = data['Inflation Rate (%)'].astype(float)
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

    # Predict future inflation rates
    future_years = pd.DataFrame({'Year': range(2025, 2031)})
    future_predictions = model.predict(future_years)

    # Display predictions
    predictions_df = pd.DataFrame({
        'Year': future_years['Year'],
        'Predicted Inflation Rate (%)': future_predictions
    })
    st.subheader("Future Predictions")
    st.write(predictions_df)

    # Visualization
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
