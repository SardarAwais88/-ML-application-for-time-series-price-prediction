# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px

# Function to train the model
def train_model(data):
    # Assume your time series data has 'Date' and 'Close' columns
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Feature engineering: extracting date-related features
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    # Split data into features and target variable
    X = data[['Year', 'Month', 'Day', 'DayOfWeek']]
    y = data['Close']  # Assuming 'Close' is the target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple Random Forest Regressor as an example
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Display evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    st.write(f'Mean Squared Error: {mse:.2f}')

    # Display a plot of actual vs predicted prices
    fig = px.scatter(x=y_test.index, y=y_test, labels={'y': 'Actual Price'}, title='Actual vs Predicted Prices')
    fig.add_trace(px.scatter(x=y_test.index, y=predictions, labels={'y': 'Predicted Price'}).data[0])
    st.plotly_chart(fig)

    return model

# Function to make predictions
def predict_price(model, input_features):
    # Assuming input_features is a dictionary containing Year, Month, Day, DayOfWeek
    input_data = pd.DataFrame([input_features])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
def main():
    st.title('Time Series Price Prediction App')

    # Upload data
    uploaded_file = st.file_uploader('Upload your time series data (CSV file)', type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Train the model
        st.header('Train the Model')
        trained_model = train_model(data)

        # Prediction
        st.header('Make Predictions')
        
        # Use st.form to create a form
        with st.form(key='prediction_form'):
            input_features = {}
            for col in ['Year', 'Month', 'Day', 'DayOfWeek']:
                # Use st.number_input to get user input
                input_features[col] = st.number_input(f'Enter {col}', value=2023)

            # Use st.form_submit_button to add a submit button
            submit_button = st.form_submit_button(label='Predict')

        # Check if the form is submitted
        if submit_button:
            # Create a dictionary to store the input features
            prediction = predict_price(trained_model, input_features)
            st.success(f'Predicted Price: {prediction:.2f}')

# Run the app
if __name__ == '__main__':
    main()
