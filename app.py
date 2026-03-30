import streamlit as st
import joblib
import json
import pandas as pd

model = joblib.load("model.pkl")

with open("columns.json", "r") as f:
    data_columns = json.load(f)

location_columns = [col for col in data_columns if col.startswith("location_")]
locations = sorted([col.replace("location_", "") for col in location_columns])

st.set_page_config(page_title="Bangalore House Price Prediction", page_icon="🏠")

st.title("🏠 Bangalore House Price Prediction")
st.write("Enter property details to estimate the house price.")

total_sqft = st.number_input("Total Square Feet", min_value=300.0, step=50.0)
bath = st.number_input("Number of Bathrooms", min_value=1, step=1)
bhk = st.number_input("Number of BHK", min_value=1, step=1)
location = st.selectbox("Select Location", locations)

if st.button("Predict Price"):
    input_data = {col: 0 for col in data_columns}
    input_data["total_sqft"] = total_sqft
    input_data["bath"] = bath
    input_data["bhk"] = bhk

    location_col = f"location_{location}"
    if location_col in input_data:
        input_data[location_col] = 1

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.success(f"Estimated House Price: ₹ {prediction:.2f} Lakhs")