# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ----------------------------
# LSTM Forecast Function
# ----------------------------
def lstm_forecast(df, target_column="Manok"):
    data = df[[target_column]].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(7, len(scaled)):
        X.append(scaled[i-7:i,0])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=1, verbose=0)

    # Forecast next 3 days
    forecast_input = scaled[-7:].reshape(1,7,1)
    forecast = []
    for _ in range(3):
        pred = model.predict(forecast_input, verbose=0)[0][0]
        forecast.append(pred)
        forecast_input = np.append(forecast_input[:,1:,:], [[pred]], axis=1)
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))
    return forecast.flatten()

# ----------------------------
# Weather Function (Pagasa placeholder)
# ----------------------------
def get_weather():
    try:
        # Placeholder for demo
        return "Weather: Sunny 30°C"
    except:
        return "Weather data unavailable"

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="🌾 Agri Forecast App", layout="wide")
st.title("🌾 Agri Price Forecast (3-Day)")

# Top bar: Weather + Current date/time
col1, col2 = st.columns([1,2])
with col1:
    st.markdown(f"**{get_weather()}**")
with col2:
    st.markdown(f"**{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    if "df" not in st.session_state:
        st.session_state.df = new_df
    else:
        st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
    st.success(f"CSV Loaded: {len(st.session_state.df)} rows")
else:
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()

# Generate Forecast
if st.button("Generate 3-Day Forecast for 'Manok'"):
    if st.session_state.df.empty:
        st.warning("Please upload CSV first.")
    else:
        forecast = lstm_forecast(st.session_state.df, target_column="Manok")
        forecast_dates = [datetime.now() + timedelta(days=i+1) for i in range(3)]

        # Plot historical + forecast
        plt.figure(figsize=(10,5))
        plt.plot(st.session_state.df["Manok"].values, label="Historical")
        plt.plot(range(len(st.session_state.df), len(st.session_state.df)+3),
                 forecast, label="Forecast", marker='o')
        plt.title("Manok Price Forecast")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

        # Show forecast table
        forecast_df = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "Forecast_Manok": forecast
        })
        st.table(forecast_df)
