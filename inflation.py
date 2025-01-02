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

        # Proceed with the rest of the logic (model training, predictions, visualization, etc.)
        X = data[['Year']]
        y = data['Inflation Rate (%)']
        
        # Split the data
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error: {mse}")

        # Future predictions
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
        import matplotlib.pyplot as plt
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

    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
