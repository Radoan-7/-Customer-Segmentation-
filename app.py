import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="centered"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        body {
            background: linear-gradient(270deg, #004d40, #00e676, #004d40);
            background-size: 600% 600%;
            animation: gradientShift 12s ease infinite;
            font-family: 'Inter', sans-serif;
            color: #f1f8e9;
            margin: 0;
        }

        .main-container {
            background: rgba(255, 255, 255, 0.08);
            padding: 50px;
            border-radius: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            backdrop-filter: blur(25px);
            max-width: 850px;
            margin: 70px auto;
            animation: fadeSlide 1.2s ease-out;
            border: 1px solid rgba(0, 230, 118, 0.5);
        }

        h1 {
            text-align: center;
            font-weight: 800;
            font-size: 3rem;
            color: #e8f5e9;
            letter-spacing: 3px;
            text-shadow: 0 2px 10px rgba(0, 230, 118, 0.7);
        }

        .description {
            text-align: center;
            font-size: 1.3rem;
            margin-bottom: 30px;
            color: #c8e6c9;
            opacity: 0.85;
        }

        .stNumberInput > div > div > input {
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid #66bb6a;
            border-radius: 15px;
            padding: 12px;
            font-size: 16px;
            color: #ffffff;
            text-align: center;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }

        .stNumberInput > div > div > input:hover {
            box-shadow: 0 0 12px rgba(0, 230, 118, 0.7);
            border-color: #00e676;
        }

        .stButton > button {
            background: linear-gradient(135deg, #00c853, #1de9b6);
            color: #004d40;
            border-radius: 15px;
            padding: 14px 30px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
            cursor: pointer;
            transition: all 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1de9b6, #00e676);
            transform: translateY(-5px) scale(1.05);
            box-shadow: 0 12px 30px rgba(0,0,0,0.5);
        }

        .prediction-box {
            margin-top: 35px;
            padding: 30px;
            border-radius: 20px;
            background: rgba(0, 230, 118, 0.25);
            text-align: center;
            font-size: 1.8rem;
            font-weight: 800;
            color: #ffffff;
            box-shadow: 0 8px 25px rgba(0,0,0,0.5);
            animation: pulse 1.2s infinite, pop 0.6s ease-in-out;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        @keyframes fadeSlide {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pop {
            0% { transform: scale(0.7); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 10px rgba(0,230,118,0.5); }
            50% { box-shadow: 0 0 25px rgba(0,230,118,0.9); }
            100% { box-shadow: 0 0 10px rgba(0,230,118,0.5); }
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1>Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Analyze customer data and discover their ideal cluster instantly.</p>", unsafe_allow_html=True)

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (Days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.markdown(f"<div class='prediction-box'>Predicted Segment: Cluster {cluster}</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
