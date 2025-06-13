# Import the required libraries: Streamlit, NumPy, and Pillow (PIL).
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import gdown
from PIL import Image

# Set the page configuration of the app
st.set_page_config(
    page_title="Timelytics: Delivery Time Predictor",
    page_icon="🚚",
    layout="centered"
)

# Display the title and captions for the app
st.title("📦 Timelytics: Optimize Your Supply Chain with Advanced Forecasting")

st.caption(
    "Timelytics is an ensemble model that utilizes XGBoost, Random Forests, and SVM "
    "to accurately forecast Order to Delivery (OTD) times. It helps businesses to identify "
    "potential delays and optimize logistics."
)

st.caption(
    "The model uses historical data like processing times, distances, and order metadata "
    "to generate forecasts, enabling better inventory management and faster deliveries."
)

# Load the trained ensemble model from Google Drive
@st.cache_resource
def load_model():
    model_path = "voting_model.pkl"
    file_id = "1FKcK2AcbdbGNOhpZLiPa4e2wM8ZSLJoY"
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    with open(model_path, "rb") as file:
        return pickle.load(file)

voting_model = load_model()

# Define the function for the wait time predictor
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    prediction = voting_model.predict(
        np.array(
            [[
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance
            ]]
        )
    )
    return round(prediction[0])

# Define the input parameters using Streamlit's sidebar
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img, use_column_width=True)
    st.header("🔢 Input Parameters")

    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", min_value=2000, max_value=2100, value=2018)
    product_size_cm3 = st.number_input("Product Size (cm³)", min_value=1, value=9328)
    product_weight_g = st.number_input("Product Weight (g)", min_value=1, value=1800)
    geolocation_state_customer = st.number_input("Customer Geolocation State Code", min_value=0, value=10)
    geolocation_state_seller = st.number_input("Seller Geolocation State Code", min_value=0, value=20)
    distance = st.number_input("Distance (km)", min_value=0.0, value=475.35)

    # Submit button
    submit = st.button("🚀 Predict Delivery Time")

# Output section
st.header("📊 Output: Predicted Delivery Time")

if submit:
    with st.spinner("Predicting..."):
        prediction = waitime_predictor(
            purchase_dow,
            purchase_month,
            year,
            product_size_cm3,
            product_weight_g,
            geolocation_state_customer,
            geolocation_state_seller,
            distance,
        )
        st.success(f"✅ Estimated Delivery Time: **{prediction} days**")

# Sample dataset for demonstration
data = {
    "Purchased Day of the Week": [0, 3, 1],
    "Purchased Month": [6, 3, 1],
    "Purchased Year": [2018, 2017, 2018],
    "Product Size in cm³": [37206.0, 63714.0, 54816.0],
    "Product Weight in grams": [16250.0, 7249.0, 9600.0],
    "Geolocation State Customer": [25, 25, 25],
    "Geolocation State Seller": [20, 7, 20],
    "Distance (km)": [247.94, 250.35, 4.915]
}
df = pd.DataFrame(data)

# Display sample data
st.header("📁 Sample Input Dataset")
st.dataframe(df)
