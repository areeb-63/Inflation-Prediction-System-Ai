import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

# Streamlit app
def main():
    st.title("Pakistan Inflation Rate Prediction")

    # Model selection
    model_option = st.selectbox("Select model", ["Linear Regression", "Polynomial Regression (Degree 2)"])

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

        # Select model based on user choice
        if model_option == "Linear Regression":
            model = LinearRegression()
        elif model_option == "Polynomial Regression (Degree 2)":
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
            model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")
        st.write(f"Mean Absolute Error: {mae}")

        # User input for selecting prediction years
        start_year = st.slider("Select starting year for prediction:", min_value=2025, max_value=2030, value=2025)
        end_year = st.slider("Select ending year for prediction:", min_value=2026, max_value=2031, value=2030)

        # Create future years for prediction
        future_years = pd.DataFrame({'Year': range(start_year, end_year + 1)})
        if model_option == "Polynomial Regression (Degree 2)":
            future_years_poly = poly.transform(future_years)
            future_predictions = model.predict(future_years_poly)
        else:
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
        plt.title(f'Inflation Rate Prediction using {model_option}')
        plt.legend()
        st.pyplot(plt)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr_matrix = data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt)

        # Feature Importance (for models that have feature importance, like RandomForest)
        if model_option == "Linear Regression":
            feature_importance = model.coef_
            feature_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            st.subheader("Feature Importance")
            st.write(feature_df)

        # Prediction Uncertainty (Confidence Interval)
        st.subheader("Prediction Uncertainty")
        # Here we calculate the confidence interval for predictions
        from sklearn.utils import resample
        n_iterations = 1000
        predictions_bootstrap = np.zeros((n_iterations, len(future_years)))
        for i in range(n_iterations):
            X_resample, y_resample = resample(X_train, y_train, random_state=i)
            model.fit(X_resample, y_resample)
            if model_option == "Polynomial Regression (Degree 2)":
                future_years_poly = poly.transform(future_years)
                predictions_bootstrap[i] = model.predict(future_years_poly)
            else:
                predictions_bootstrap[i] = model.predict(future_years)
        
        # Calculate 95% Confidence Interval for future predictions
        lower_bound = np.percentile(predictions_bootstrap, 2.5, axis=0)
        upper_bound = np.percentile(predictions_bootstrap, 97.5, axis=0)
        ci_df = pd.DataFrame({
            'Year': future_years['Year'],
            'Lower Bound (95% CI)': lower_bound,
            'Upper Bound (95% CI)': upper_bound
        })
        st.write(ci_df)

    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
