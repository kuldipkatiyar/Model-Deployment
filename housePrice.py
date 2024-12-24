import streamlit as st
import pandas as pd
import pickle

# Load the trained model (replace 'california_housing_model.pkl' with your model file)
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the features used in the model (replace with your actual features)
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# Create the Streamlit app
st.title('California Housing Price Prediction')

# Get user input for each feature
user_input = {}
for feature in features:
    if feature in ['Latitude', 'Longitude']:
        user_input[feature] = st.number_input(feature, value=37.0, min_value=32.0, max_value=42.0, step=0.1)
    else:
        user_input[feature] = st.number_input(feature, value=0.0)

# Create a DataFrame from user input
input_df = pd.DataFrame([user_input])

# Make prediction when the user clicks the button
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.success(f'Predicted Median House Value: ${prediction[0]:.2f}')