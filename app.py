# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import os
import requests
from io import StringIO
import time

# -------------------------------
# 1. Page Setup
# -------------------------------
st.set_page_config(page_title="🌾 AI Price Forecast", layout="wide")

# Display current date and time at top right
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>🤖 SANDRA MAE P. DELFINO - Agricultural Products Price and Demand Prediction for Small-Scale Famers in Partido, Camarines Sur</h1>
        </div>
        <div style="text-align: right;">
            <p>{current_time}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
This app uses an **LSTM neural network** to predict prices of key agricultural commodities.
""")

# Output folder
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# Weather Forecast Section
# -------------------------------
st.sidebar.header("🌤️ Weather Forecast")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_weather_forecast():
    try:
        # This is a placeholder for actual PAGASA API integration
        # In a real implementation, you would use the PAGASA API or web scraping
        # For demonstration, we'll return sample data
        
        # Simulate API delay
        time.sleep(0.5)
        
        # Sample weather data (replace with actual API call)
        weather_data = {
            "today": {"temp": "28°C", "condition": "Partly cloudy", "rain": "30%"},
            "tomorrow": {"temp": "29°C", "condition": "Scattered thunderstorms", "rain": "60%"},
            "day_after": {"temp": "27°C", "condition": "Rainy", "rain": "80%"}
        }
        
        return weather_data
    except:
        return None

weather = get_weather_forecast()

if weather:
    st.sidebar.subheader("Today")
    st.sidebar.write(f"Temperature: {weather['today']['temp']}")
    st.sidebar.write(f"Condition: {weather['today']['condition']}")
    st.sidebar.write(f"Chance of rain: {weather['today']['rain']}")
    
    st.sidebar.subheader("Tomorrow")
    st.sidebar.write(f"Temperature: {weather['tomorrow']['temp']}")
    st.sidebar.write(f"Condition: {weather['tomorrow']['condition']}")
    st.sidebar.write(f"Chance of rain: {weather['tomorrow']['rain']}")
    
    st.sidebar.subheader("Day After Tomorrow")
    st.sidebar.write(f"Temperature: {weather['day_after']['temp']}")
    st.sidebar.write(f"Condition: {weather['day_after']['condition']}")
    st.sidebar.write(f"Chance of rain: {weather['day_after']['rain']}")
else:
    st.sidebar.error("Could not fetch weather data. Please try again later.")

# -------------------------------
# CSV Upload Functionality
# -------------------------------
st.sidebar.header("📤 Upload New Data")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def load_data():
    # Read the CSV content directly from the provided text
    content = """Date,Manok,Baboy,Baka,Telapia,Bangus,Galunggong,Tamban,Tinapa,Repolyo,Kamote,Kalabasa,Patatas,Niyog,Saging,Sugar,Milled,Sibuyas,Kamatis,Bawang,Itlog,Mantika,Ampalaya,Talong,Manga,Pinya,Calamansi,Rainfall,Holiday,Demand Index
01-Aug-25,176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,0,50
02-Aug-25,178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,346,41,45,178,10,0,52
03-Aug-25,180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53
04-Aug-25,181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54
05-Aug-25,182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56
06-Aug-25,180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55
07-Aug-25,179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57
08-Aug-25,181,355,181,355,181,355,181,355,77,38,44,181,355,181,355,355,77,38,44,181,355,181,355,38,44,181,20,0,58
09-Aug-25,183,356,183,356,183,356,183,356,78,39,45,183,356,183,356,356,78,39,45,183,356,183,356,39,45,183,2,1,80
10-Aug-25,184,354,184,354,184,354,184,354,79,40,45,184,354,184,354,354,79,40,45,184,354,184,354,40,45,184,0,1,85
11-Aug-25,176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,极,50
12-Aug-25,178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,346,41,45,178,10,0,52
13-Aug-25,180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,极,0,0,53
14-Aug-25,181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54
15-Aug-25,182,350,182,350,182,350,182,350,80,41,45,182,350,182,极,350,80,41,45,182,350,182,350,41,45,182,7,0,56
16-Aug-25,180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55
17-Aug-25,179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57
18-Aug-25,200,300,200,300,200,300,200,300,79,40,44,200,300,200,300,300,79,40,44,200,300,200,300,40,44,200,5,0,50
19-Aug-25,150,320,150,320,150,320,150,320,80,41,45,150,320,150,320,320,80,41,45,150,320,150,320,41,45,150,10,0,52
20-Aug-25,145,极,145,310,145,310,145,310,81,41,46,145,310,145,310,310,81,41,46,145,310,145,310,41,46,145,0,0,53
21-Aug-25,185,290,185,290,185,290,185,290,82,42,46,185,290,185,290,290,82,42,46,185,290,185,290,42,46,185,3,0,54
22-Aug-25,182,305,182,305,182,305,182,305,80,41,45,182,305,182,305,305,80,41,45,182,305,182,305,41,45,182,7,0,56
23-Aug-25,180,315,180,315,180,315,180,315,79,40,44,180,315,180,315,315,79,极,44,180,315,180,315,40,44,180,12,0,55
24-Aug-25,179,310,179,310,179,310,179,310,78,39,43,179,310,179,310,310,78,39,43,179,310,179,310,39,43,179,15,0,57
25-Aug-25,181,305,181,305,181,305,181,305,77,38,44,181,305,181,305,305,77,38,44,181,305,181,305,38,44,181,20,0,58
01-Jul-25,176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,极,345,176,345,40,44,176,5,0,50
02-Jul-25,178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,极,80,41,45,178,346,178,346,41,45,178,10,0,52
03-Jul-25,180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53
04-Jul-25,181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54
05-Jul-25,182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56
06-Jul-25,180,352,180,352,180,352,极,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55
07-Jul-25,179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57
08-Jul-25,181,355,181,355,181,355,181,355,77,38,44,181,355,181,355,355,77,38,44,181,355,181,355,38,44,181,20,0,58
09-Jul-25,183,356,183,356,183,356,183,356,78,39,45,183,356,183,356,356,78,39,45,183,356,183,356,39,45,183,2,1,80
10-Jul-25,184,354,184,354,184,354,184,354,79,40,45,184,354,184,354,354,79,40,45,184,354,184,354,40,45,184,0,1,85
11-Jul-25,176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,0,50
12-Jul-25,178,346,178,346,178,346,178,346,极,41,45,178,346,178,极,346,80,41,45,178,346,178,346,41,45,178,10,0,52
13-Jul-25,180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53
14-Jul-25,181,349,极,349,181,349,181,349,82,42,极,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54
15-Jul-25,182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56
16-Jul-25,180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55
17-Jul-25,179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57
18-Jul-25,200,300,200,300,200,300,200,300,79,40,44,200,300,200,300,300,79,40,44,200,300,200,300,40,44,200,20,0,58
19-Jul-25,150,320,150,320,150,320,150,320,80,41,45,150,320,150,320,320,80,41,45,150,极,150,320,41,45,150,2,1,80
20-Jul-25,145,310,145,310,145,310,145,310,81,41,46,145,310,145,310,310,81,41,46,145,310,145,310,41,46,145,0,1,85
21-Jul-25,185,290,185,290,185,290,185,290,82,42,46,185,290,185,290,290,82,42,46,185,290,185,290,42,46,185,5,0,50
22-Jul-25,182,305,182,305,182,305,182,305,80,41,45,182,305,182,305,305,80,41,45,182,305,182,305,41,45,182,10,0,52
23-Jul-25,180,315,180,315,180,315,180,315,79,40,44,180,315,180,315,315,79,40,44,180,315,180,315,40,44,180,0,0,53
24-Jul-25,179,310,179,310,179,310,179,310,78,39,43,179,310,179,310,310,78,39,43,179,310,179,310,39,43,179,3,0,54
25-Jul-25,181,305,181,305,181,305,181,305,77,38,44,181,305,181,305,305,77,38,44,181,305,181,305,38,44,181,7,0,56
26-Jul-25,180,352,182,305,182,305,182,305,79,40,44,180,352,182,305,182,305,182,305,79,40,44,180,352,182,305,12,0,55
27-Jul极,179,353,180,315,180,315,180,315,78,39,43,179,353,180,315,180,315,180,315,78,39,43,179,353,180,315,15,0,57
28-Jul-25,181,355,179,310,179,310,179,极,77,38,44,181,355,179,310,179,310,179,310,77,38,44,181,355,179,310,20,0,58
29-Jul-25,183,356,181,305,181,305,181,305,78,39,45,183,356,181,305,181,305,181,305,78,39,45,183,356,181,305,2,1,80
30-Jul-25,184,354,179,310,179,310,179,310,79,40,45,184,354,179,310,179,310,179,310,79,40,45,184,354,179,310,0,1,85
31-Jul-25,184,354,181,305,181,305,181,305,79,40,45,184,354,181,305,181,305,181,305,79,40,45,184,354,181,305,0,1,85
01-Jun-25,176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,极,176,5,0,50
02-Jun-25,178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,极,41,45,178,10,0,52
03-Jun-25,180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53
04-Jun-25,181,349,181,349,极,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54
05-Jun-25,182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56
06-Jun-25,180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55
07-Jun-25,179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57
08-Jun-25,181,355,181,355,181,355,181,355,77,38,44,181,355,181,355,355,77,38,44,181,355,181,355,38,44,181,20,0,58
09-Jun-25,183,356,183,356,183,356,183,极,78,39,极,183,356,183,356,356,78,39,45,183,356,183,356,39,45,183,2,1,80
10-Jun-25,184,354,184,354,184,354,184,354,79,40,45,184,354,184,354,354,79,40,45,184,354,184,354,40,45,184,0,1,85
11-Jun-25,176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,0,50
12-Jun-25,178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,346,41,45,178,10,0,52
13-Jun-25,180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,极,41,46,180,347,180,347,41,46,180,0,0,53
14-Jun-25,181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,极
15-Jun-25,182,350,182,350,极,350,182,350,80,41,45,182,极,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56
16-Jun-25,180,352,180,352,180,352,180,352,79,极,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55
17-Jun-25,179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57
18-Jun-25,200,300,200,300,200,300,200,300,79,40,44,200,300,200,300,300,79,40,44,200,300,200,300,40,44,200,5,0,50
19-Jun-25,150,320,150,320,150,320,150,320,80,41,45,150,320,极,320,320,80,41,45,150,320,150,320,41,45,极,10,0,52
20-Jun-25,145,310,145,310,145,310,145,310,81,41,46,145,310,145,310,310,81,41,46,145,310,145,310,41,46,145,0,0,53
21-Jun-25,185,290,185,290,185,290,185,290,82,42,46,185,290,185,290,290,82,42,46,185,290,185,290,42,46,185,3,0,54
22-Jun-25,182,305,182,305,182,305,182,305,80,41,45,182,305,182,305,305,80,41,45,182,305,182,305,41,45,182,7,0,56
23-Jun-25,180,315,180,315,180,315,180,315,79,40,44,180,315,180,315,315,79,40,44,180,315,180,315,40,44,180,12,0,55
24-Jun-25,179,310,179,310,179,310,179,310,78,39,43,179,310,179,310,310,78,39,43,179,310,179,310,39,43,179,15,0,57
25-Jun-25,181,305,181,305,181,305,181,305,77,38,44,181,305,181,305,305,77,38,44,181,305,181,305,38,44,181,20,0,58
26-Jun-25,185,290,185,290,185,290,185,290,82,42,46,185,290,185,290,290,82,42,46,185,290,185,290,42,46,185,3,0,54
27-Jun-25,182,305,182,305,182,305,182,极,80,41,45,182,305,182,305,305,80,41,45,182,305,182,305,41,45,182,7,0,56
28-Jun-25,180,315,180,315,180,315,180,315,79,40,44,180,315,180,315,315,79,40,44,180,315,180,315,40,44,180,12,0,55
29-Jun-25,179,310,179,310,179,310,179,310,78,39,43,179,310,179,310,310,78,39,43,179,310,179,310,39,43,179,15,0,57
30-Jun-25,181,305,181,305,181,305,181,305,77,38,44,181,305,181,305,305,77,38,44,181,305,181,305,38,44,181,20,0,58"""

    # Create DataFrame from the content
    df = pd.read_csv(StringIO(content))
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Date'])
    
    # Set Date as index and sort
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    return df

# Load initial data
df = load_data()

# Handle CSV upload
if uploaded_file is not None:
    try:
        # Read uploaded CSV
        new_data = pd.read_csv(uploaded_file)
        
        # Convert Date column to datetime
        new_data['Date'] = pd.to_datetime(new_data['Date'], dayfirst=True, errors='coerce')
        
        # Drop rows with invalid dates
        new_data = new_data.dropna(subset=['Date'])
        
        # Set Date as index
        new_data.set_index('Date', inplace=True)
        
        # Combine with existing data, keeping the latest values for overlapping dates
        df = pd.concat([df, new_data]).groupby('Date').last().sort_index()
        
        st.sidebar.success("✅ New data successfully integrated!")
        
    except Exception as e:
        st.sidebar.error(f"Error processing uploaded file: {str(e)}")

# Define features and targets
features = ["Manok", "Baboy", "Baka", "Telapia", "Bangus", "Galunggong", "Tamban", "Tinapa", "Repolyo", "Kamote", "Kalabasa", "Patatas", "Niyog", "Saging", "Sugar", "Milled", "Sibuyas", "Kamatis", "Bawang", "Itlog", "Mantika", "Ampalaya", "Talong", "Manga", "Pinya", "Calamansi", "Rainfall", "Holiday", "Demand Index"]
targets = ["Manok", "Baboy", "Baka", "Telapia", "Bangus", "Galunggong", "Tamban", "Tinapa", "Repolyo", "Kamote", "Kalabasa", "Patatas", "Niyog", "Saging", "Sugar", "Milled", "Sibuyas", "Kamatis", "Bawang", "Itlog", "Mantika", "Ampalaya", "Talong", "Manga", "Pinya", "Calamansi"]

# Ensure all columns are numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with missing values
df = df.dropna()

st.success("✅ Data loaded and cleaned!")
st.write("### Last 10 Days of Data")
st.dataframe(df.tail(10))

# -------------------------------
# 3. Data Visualization
# -------------------------------
st.write("### 📈 Price Trends")

# Select a product to visualize
selected_product = st.selectbox("Select a product to visualize:", targets)

# Plot the selected product's price history
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df.index, df[selected_product], marker='o', linestyle='-')
ax.set_title(f'{selected_product} Price History')
ax.set_xlabel('Date')
ax.set_ylabel('Price (₱)')
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------------
# 4. Normalize Data
# -------------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

# -------------------------------
# 5. Create Sequences for LSTM
# -------------------------------
def create_sequences(data, seq_length=7):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][:len(targets)])  # Only predict the target columns
    return np.array(X), np.array(y)

SEQ_LEN = 7
X, y = create_sequences(scaled, SEQ_LEN)

# Split the data
split = max(1, int(len(X) * 0.8))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# 6. Build and Train LSTM Model
# -------------------------------
with st.spinner("🧠 Training LSTM model..."):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(len(targets))
    ])
    
    model.compile(optimizer="adam", loss="mse")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_split=0.1 if len(X_train) > 10 else 0.0,
        verbose=0
    )
    
    # Make predictions
    if len(X_test) > 0:
        preds = model.predict(X_test, verbose=0)
    else:
        preds = np.array([])

# Function to rescale predictions
def rescale_predictions(preds_arr, y_test_arr):
    if preds_arr.size == 0:
        return np.array([]), np.array([])
    
    # Create full arrays with zeros for non-target features
    full_preds = np.zeros((preds_arr.shape[0], len(features)))
    full_preds[:, :len(targets)] = preds_arr
    
    full_y_test = np.zeros((y_test_arr.shape[0], len(features)))
    full_y_test[:, :len(targets)] = y_test_arr
    
    # Inverse transform
    preds_rescaled = scaler.inverse_transform(full_preds)[:, :len(targets)]
    y_test_rescaled = scaler.inverse_transform(full_y_test)[:, :len(targets)]
    
    return preds_rescaled, y_test_rescaled

# Rescale predictions
preds_rescaled, y_test_rescaled = rescale_predictions(preds, y_test)

st.success("✅ Model trained and predictions generated!")

# -------------------------------
# 7. Show Metrics
# -------------------------------
if preds_rescaled.size > 0:
    mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
    r2 = r2_score(y_test_rescaled, preds_rescaled)
else:
    mae = np.nan
    r2 = np.nan

st.write("### 📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (MAE)", f"₱{mae:.2f}" if not np.isnan(mae) else "N/A")
col2.metric("R² Score", f"{r2:.3f}" if not np.isnan(r2) else "N/A")

# Plot training history if history is not None
if history is not None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Model Training History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# -------------------------------
# 8. Forecast Next Day
# -------------------------------
st.write("### 🔮 Forecast Next Day")

# Get the last sequence
last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)

# Make prediction
next_pred = model.predict(last_seq, verbose=0)

# Create input controls for external factors
col1, col2, col3 = st.columns(3)
with col1:
    rainfall_val = st.slider("Rainfall", min_value=0, max_value=100, value=5)
with col2:
    holiday_val = st.selectbox("Holiday", options=[0, 1], index=0)
with col3:
    demand_val = st.slider("Demand Index", min_value=0, max_value=100, value=50)

# Create the full feature vector for inverse scaling
new_row = np.zeros((1, len(features)))
new_row[0, :len(targets)] = next_pred[0]
new_row[0, features.index("Rainfall")] = rainfall_val
new_row[0, features.index("Holiday")] = holiday_val
new_row[0, features.index("Demand Index")] = demand_val

# Inverse transform to get actual prices
forecast_price = scaler.inverse_transform(new_row)[0][:len(targets)]

# Create DataFrame for forecast
tomorrow = df.index[-1] + timedelta(days=1)
forecast_df = pd.DataFrame({
    "Product": targets,
    "Current Price (₱)": [df[target].iloc[-1] for target in targets],
    "Predicted Price (₱)": forecast_price,
    "Change (%)": [((forecast_price[i] - df[target].iloc[-1]) / df[target].iloc[-1] * 100) for i, target in enumerate(targets)]
})

st.write(f"### Forecast for {tomorrow.strftime('%Y-%m-%d')}")
st.dataframe(forecast_df.style.format({
    "Current Price (₱)": "{:.2f}",
    "Predicted Price (₱)": "{:.2f}",
    "Change (%)": "{:.2f}%"
}))

# -------------------------------
# 9. Three-Day Forecast
# -------------------------------
st.write("### 📅 Three-Day Forecast")

# Function to generate multi-day forecast
def generate_forecast(days=3):
    forecasts = []
    current_seq = scaled[-SEQ_LEN:].copy()
    
    for day in range(days):
        # Reshape for prediction
        input_seq = current_seq.reshape(1, SEQ_LEN, -1)
        
        # Make prediction
        pred = model.predict(input_seq, verbose=0)[0]
        
        # Create full feature vector
        full_pred = np.zeros(len(features))
        full_pred[:len(targets)] = pred
        # Keep external factors from the last available data
        full_pred[len(targets):] = scaled[-1, len(targets):]
        
        # Update the sequence
        current_seq = np.vstack([current_seq[1:], full_pred])
        
        # Inverse transform to get actual prices
        forecast_prices = scaler.inverse_transform(full_pred.reshape(1, -1))[0][:len(targets)]
        
        # Store forecast
        forecast_date = df.index[-1] + timedelta(days=day+1)
        forecasts.append({
            "Date": forecast_date,
            "Prices": forecast_prices
        })
    
    return forecasts

# Generate 3-day forecast
three_day_forecast = generate_forecast(3)

# Display 3-day forecast
for i, forecast in enumerate(three_day_forecast):
    st.write(f"#### {forecast['Date'].strftime('%Y-%m-%d')} (Day {i+1})")
    
    day_forecast_df = pd.DataFrame({
        "Product": targets,
        "Predicted Price (₱)": forecast['Prices']
    })
    
    st.dataframe(day_forecast_df.style.format({
        "Predicted Price (₱)": "{:.2f}"
    }))

# -------------------------------
# 10. Download Forecast
# -------------------------------
@st.cache_data
def convert_df_to_csv(_df):
    return _df.to_csv(index=False).encode('utf-8')

csv_bytes = convert_df_to_csv(forecast_df)
st.download_button(
    label="📥 Download Forecast as CSV",
    data=csv_bytes,
    file_name=f"forecast_{tomorrow.strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# -------------------------------
# 11. Show Raw Data (Optional)
# -------------------------------
if st.checkbox("Show raw data"):
    st.write("### Raw Data")
    st.dataframe(df)