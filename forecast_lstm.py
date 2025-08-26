# forecast_lstm_complete.py
# Forecast LSTM with tomorrow forecast, manual override, and author
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from datetime import datetime, timedelta
from io import BytesIO

# -------------------------------
# 1. Load and clean dataset
# -------------------------------
try:
    # Provided data
    data = [
        ["01-Aug-25",176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,0,50],
        ["02-Aug-25",178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,346,41,45,178,10,0,52],
        ["03-Aug-25",180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53],
        ["04-Aug-25",181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54],
        ["05-Aug-25",182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56],
        ["06-Aug-25",180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55],
        ["07-Aug-25",179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57],
        ["08-Aug-25",181,355,181,355,181,355,181,355,77,38,44,181,355,181,355,355,77,38,44,181,355,181,355,38,44,181,20,0,58],
        ["09-Aug-25",183,356,183,356,183,356,183,356,78,39,45,183,356,183,356,356,78,39,45,183,356,183,356,39,45,183,2,1,80],
        ["10-Aug-25",184,354,184,354,184,354,184,354,79,40,45,184,354,184,354,354,79,40,45,184,354,184,354,40,45,184,0,1,85],
        ["11-Aug-25",176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,0,50],
        ["12-Aug-25",178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,346,41,45,178,10,0,52],
        ["13-Aug-25",180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53],
        ["14-Aug-25",181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54],
        ["15-Aug-25",182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56],
        ["16-Aug-25",180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55],
        ["17-Aug-25",179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57],
        ["18-Aug-25",200,300,200,300,200,300,200,300,79,40,44,200,300,200,300,300,79,40,44,200,300,200,300,40,44,200,5,0,50],
        ["19-Aug-25",150,320,150,320,150,320,150,320,80,41,45,150,320,150,320,320,80,41,45,150,320,150,320,41,45,150,10,0,52],
        ["20-Aug-25",145,310,145,310,145,310,145,310,81,41,46,145,310,145,310,310,81,41,46,145,310,145,310,41,46,145,0,0,53],
        ["21-Aug-25",185,290,185,290,185,290,185,290,82,42,46,185,290,185,290,290,82,42,46,185,290,185,290,42,46,185,3,0,54],
        ["22-Aug-25",182,305,182,305,182,305,182,305,80,41,45,182,305,182,305,305,80,41,45,182,305,182,305,41,45,182,7,0,56],
        ["23-Aug-25",180,315,180,315,180,315,180,315,79,40,44,180,315,180,315,315,79,40,44,180,315,180,315,40,44,180,12,0,55],
        ["24-Aug-25",179,310,179,310,179,310,179,310,78,39,43,179,310,179,310,310,78,39,43,179,310,179,310,39,43,179,15,0,57],
        ["25-Aug-25",181,305,181,305,181,305,181,305,77,38,44,181,305,181,305,305,77,38,44,181,305,181,305,38,44,181,20,0,58],
        ["01-Jul-25",176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,0,50],
        ["02-Jul-25",178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,346,41,45,178,10,0,52],
        ["03-Jul-25",180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53],
        ["04-Jul-25",181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54],
        ["05-Jul-25",182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56],
        ["06-Jul-25",180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55],
        ["07-Jul-25",179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57],
        ["08-Jul-25",181,355,181,355,181,355,181,355,77,38,44,181,355,181,355,355,77,38,44,181,355,181,355,38,44,181,20,0,58],
        ["09-Jul-25",183,356,183,356,183,356,183,356,78,39,45,183,356,183,356,356,78,39,45,183,356,183,356,39,45,183,2,1,80],
        ["10-Jul-25",184,354,184,354,184,354,184,354,79,40,45,184,354,184,354,354,79,40,45,184,354,184,354,40,45,184,0,1,85],
        ["11-Jul-25",176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,0,50],
        ["12-Jul-25",178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,346,41,45,178,10,0,52],
        ["13-Jul-25",180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53],
        ["14-Jul-25",181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54],
        ["15-Jul-25",182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56],
        ["16-Jul-25",180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55],
        ["17-Jul-25",179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57],
        ["18-Jul-25",200,300,200,300,200,300,200,300,79,40,44,200,300,200,300,300,79,40,44,200,300,200,300,20,0,58],
        ["19-Jul-25",150,320,150,320,150,320,150,320,80,41,45,150,320,150,320,320,80,41,45,150,320,150,320,2,1,80],
        ["20-Jul-25",145,310,145,310,145,310,145,310,81,41,46,145,310,145,310,310,81,41,46,145,310,145,310,0,1,85],
        ["21-Jul-25",185,290,185,290,185,290,185,290,82,42,46,185,290,185,290,290,82,42,46,185,290,185,290,5,0,50],
        ["22-Jul-25",182,305,182,305,182,305,182,305,80,41,45,182,305,182,305,305,80,41,45,182,305,182,305,10,0,52],
        ["23-Jul-25",180,315,180,315,180,315,180,315,79,40,44,180,315,180,315,315,79,40,44,180,315,180,315,0,0,53],
        ["24-Jul-25",179,310,179,310,179,310,179,310,78,39,43,179,310,179,310,310,78,39,43,179,310,179,310,3,0,54],
        ["25-Jul-25",181,305,181,305,181,305,181,305,77,38,44,181,305,181,305,305,77,38,44,181,305,181,305,7,0,56],
        ["26-Jul-25",180,352,182,305,182,305,182,305,79,40,44,180,352,182,305,182,305,182,305,79,40,44,180,352,182,305,12,0,55],
        ["27-Jul-25",179,353,180,315,180,315,180,315,78,39,43,179,353,180,315,180,315,180,315,78,39,43,179,353,180,315,15,0,57],
        ["28-Jul-25",181,355,179,310,179,310,179,310,77,38,44,181,355,179,310,179,310,179,310,77,38,44,181,355,179,310,20,0,58],
        ["29-Jul-25",183,356,181,305,181,305,181,305,78,39,45,183,356,181,305,181,305,181,305,78,39,45,183,356,181,305,2,1,80],
        ["30-Jul-25",184,354,179,310,179,310,179,310,79,40,45,184,354,179,310,179,310,179,310,79,40,45,184,354,179,310,0,1,85],
        ["31-Jul-25",184,354,181,305,181,305,181,305,79,40,45,184,354,181,305,181,305,181,305,79,40,45,184,354,181,305,0,1,85],
        ["01-Jun-25",176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,0,50],
        ["02-Jun-25",178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,346,41,45,178,10,0,52],
        ["03-Jun-25",180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53],
        ["04-Jun-25",181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54],
        ["05-Jun-25",182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56],
        ["06-Jun-25",180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55],
        ["07-Jun-25",179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57],
        ["08-Jun-25",181,355,181,355,181,355,181,355,77,38,44,181,355,181,355,355,77,38,44,181,355,181,355,38,44,181,20,0,58],
        ["09-Jun-25",183,356,183,356,183,356,183,356,78,39,45,183,356,183,356,356,78,39,45,183,356,183,356,39,45,183,2,1,80],
        ["10-Jun-25",184,354,184,354,184,354,184,354,79,40,45,184,354,184,354,354,79,40,45,184,354,184,354,40,45,184,0,1,85],
        ["11-Jun-25",176,345,176,345,176,345,176,345,79,40,44,176,345,176,345,345,79,40,44,176,345,176,345,40,44,176,5,0,50],
        ["12-Jun-25",178,346,178,346,178,346,178,346,80,41,45,178,346,178,346,346,80,41,45,178,346,178,346,41,45,178,10,0,52],
        ["13-Jun-25",180,347,180,347,180,347,180,347,81,41,46,180,347,180,347,347,81,41,46,180,347,180,347,41,46,180,0,0,53],
        ["14-Jun-25",181,349,181,349,181,349,181,349,82,42,46,181,349,181,349,349,82,42,46,181,349,181,349,42,46,181,3,0,54],
        ["15-Jun-25",182,350,182,350,182,350,182,350,80,41,45,182,350,182,350,350,80,41,45,182,350,182,350,41,45,182,7,0,56],
        ["16-Jun-25",180,352,180,352,180,352,180,352,79,40,44,180,352,180,352,352,79,40,44,180,352,180,352,40,44,180,12,0,55],
        ["17-Jun-25",179,353,179,353,179,353,179,353,78,39,43,179,353,179,353,353,78,39,43,179,353,179,353,39,43,179,15,0,57],
        ["18-Jun-25",200,300,200,300,200,300,200,300,79,40,44,200,300,200,300,300,79,40,44,200,300,200,300,40,44,200,5,0,50],
        ["19-Jun-25",150,320,150,320,150,320,150,320,80,41,45,150,320,150,320,320,80,41,45,150,320,150,320,41,45,150,10,0,52],
        ["20-Jun-25",145,310,145,310,145,310,145,310,81,41,46,145,310,145,310,310,81,41,46,145,310,145,310,41,46,145,0,0,53],
        ["21-Jun-25",185,290,185,290,185,290,185,290,82,42,46,185,290,185,290,290,82,42,46,185,290,185,290,42,46,185,3,0,54],
        ["22-Jun-25",182,305,182,305,182,305,182,305,80,41,45,182,305,182,305,305,80,41,45,182,305,182,305,41,45,182,7,0,56],
        ["23-Jun-25",180,315,180,315,180,315,180,315,79,40,44,180,315,180,315,315,79,40,44,180,315,180,315,40,44,180,12,0,55],
        ["24-Jun-25",179,310,179,310,179,310,179,310,78,39,43,179,310,179,310,310,78,39,43,179,310,179,310,39,43,179,15,0,57],
        ["25-Jun-25",181,305,181,305,181,305,181,305,77,38,44,181,305,181,305,305,77,38,44,181,305,181,305,38,44,181,20,0,58],
        ["26-Jun-25",185,290,185,290,185,290,185,290,82,42,46,185,290,185,290,290,82,42,46,185,290,185,290,42,46,185,3,0,54],
        ["27-Jun-25",182,305,182,305,182,305,182,305,80,41,45,182,305,182,305,305,80,41,45,182,305,182,305,41,45,182,7,0,56],
        ["28-Jun-25",180,315,180,315,180,315,180,315,79,40,44,180,315,180,315,315,79,40,44,180,315,180,315,40,44,180,12,0,55],
        ["29-Jun-25",179,310,179,310,179,310,179,310,78,39,43,179,310,179,310,310,78,39,43,179,310,179,310,39,43,179,15,0,57],
        ["30-Jun-25",181,305,181,305,181,305,181,305,77,38,44,181,305,181,305,305,77,38,44,181,305,181,305,38,44,181,20,0,58]
    ]

    columns = [
        "Date","Manok","Baboy","Baka","Telapia","Bangus","Galunggong","Tamban","Tinapa",
        "Repolyo","Kamote","Kalabasa","Patatas","Niyog","Saging","Sugar","Milled","Sibuyas",
        "Kamatis","Bawang","Itlog","Mantika","Ampalaya","Talong","Manga","Pinya","Calamansi",
        "Rainfall","Holiday","DemandIndex"
    ]

    df = pd.DataFrame(data, columns=columns)

    # Date parsing
    def parse_dates(series):
        try:
            return pd.to_datetime(series, dayfirst=True, errors='coerce')
        except:
            return pd.to_datetime(series, errors='coerce')

    df['Date'] = parse_dates(df['Date'])
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    print("Data loaded successfully:")
    print(df.tail(10))

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# -------------------------------
# 2. Define features and targets (ALL items are both)
# -------------------------------
features = [
    "Manok","Baboy","Baka","Telapia","Bangus","Galunggong","Tamban","Tinapa",
    "Repolyo","Kamote","Kalabasa","Patatas","Niyog","Saging","Sugar","Milled",
    "Sibuyas","Kamatis","Bawang","Itlog","Mantika","Ampalaya","Talong","Manga",
    "Pinya","Calamansi","Rainfall","Holiday","DemandIndex"
]

# All columns are also forecast targets according to your instruction
targets = features.copy()

# Add missing numeric columns with sensible defaults
defaults = {
    "Rainfall": 5,
    "Holiday": 0,
    "DemandIndex": 60
}
for col in features:
    if col not in df.columns:
        print(f"Column '{col}' missing. Adding default values.")
        df[col] = defaults.get(col, 0)

# Ensure numeric types for all features
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=features, inplace=True)

# -------------------------------
# 3. Normalize data
# -------------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
print(f"Data normalized. Shape: {scaled.shape}")

# -------------------------------
# 4. Create sequences (7-day lookback)
# -------------------------------
def create_sequences(data, seq_length=7, n_targets=None):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        # full vector of targets (here targets == features)
        y.append(data[i + seq_length][:n_targets])
    return np.array(X), np.array(y)

SEQ_LEN = 7
N_FEATURES = len(features)
X, y = create_sequences(scaled, SEQ_LEN, n_targets=len(targets))
print(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

# -------------------------------
# 5. Train/test split
# -------------------------------
split = max(1, int(len(X) * 0.8))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train/Test split: {len(X_train)} train, {len(X_test)} test samples")

# -------------------------------
# 6. Build LSTM model
# -------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(len(targets))
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# -------------------------------
# 7. Train model
# -------------------------------
print("Training model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=8,
                    validation_split=0.1 if len(X_train) > 10 else 0.0, verbose=1)

# -------------------------------
# 8. Make predictions
# -------------------------------
preds = model.predict(X_test) if len(X_test) > 0 else np.array([])

def rescale_predictions(preds, y_test, scaler, n_features, n_targets):
    if preds.size == 0:
        return np.array([]), np.array([])
    # pad with zeros to reconstruct full feature vector for inverse_transform
    preds_full = np.hstack([preds, np.zeros((preds.shape[0], n_features - n_targets))])
    y_test_full = np.hstack([y_test, np.zeros((y_test.shape[0], n_features - n_targets))])
    preds_rescaled = scaler.inverse_transform(preds_full)[:, :n_targets]
    y_test_rescaled = scaler.inverse_transform(y_test_full)[:, :n_targets]
    return preds_rescaled, y_test_rescaled

preds_rescaled, y_test_rescaled = rescale_predictions(preds, y_test, scaler, N_FEATURES, len(targets))

# -------------------------------
# 9. Bar Graph: Last 5 Predicted Days (plot subset if many targets)
# -------------------------------
os.makedirs("outputs", exist_ok=True)
if preds_rescaled.size > 0 and len(y_test_rescaled) >= 5:
    test_dates = df.index[SEQ_LEN + split:]
    last_5_indices = slice(-5, None)
    actual_last_5 = y_test_rescaled[last_5_indices]
    pred_last_5 = preds_rescaled[last_5_indices]
    date_labels = [d.strftime("%b %d") for d in test_dates[last_5_indices]]
    x = np.arange(len(date_labels))

    # Choose up to 6 targets to visualize (keeps plot readable)
    num_plots = min(6, len(targets))
    plt.figure(figsize=(16, 10))
    for i in range(num_plots):
        col = targets[i]
        plt.subplot(2, 3, i + 1)
        width = 0.35
        plt.bar(x - width/2, actual_last_5[:, i], width, label="Actual")
        plt.bar(x + width/2, pred_last_5[:, i], width, label="Predicted")
        plt.title(f"{col} - Last 5 Days", fontsize=10)
        plt.ylabel("Price / Value")
        plt.xlabel("Date")
        plt.xticks(x, date_labels)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
    # Author name
    plt.figtext(0.9, 0.02, "BERNIESORIANO", fontsize=8, ha="right")
    plt.tight_layout()
    chart_path = "outputs/forecast_last_5_days.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Bar chart saved to: {chart_path}")
    plt.show()
else:
    print("Not enough data to plot last 5 days comparison.")

# -------------------------------
# 10. Print evaluation metrics
# -------------------------------
if preds_rescaled.size > 0:
    mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, preds_rescaled))
    r2 = r2_score(y_test_rescaled, preds_rescaled)

    print("\nModel Performance (Overall):")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
else:
    print("No predictions to evaluate.")

# -------------------------------
# 11. Forecast for Tomorrow (Manual Override)
# -------------------------------
today = df.index[-1].date()
tomorrow = today + timedelta(days=1)

# Ask user for manual override (interactive)
try:
    manual_input = input("Mark tomorrow as holiday? (y/n, default=n): ").strip().lower()
except Exception:
    manual_input = "n"
holiday_val = 1 if manual_input == "y" else defaults.get('Holiday', 0)

demand_val = defaults.get('DemandIndex', 60)
rainfall_val = defaults.get('Rainfall', 5)

# Optional: ask user to customize DemandIndex & Rainfall
try:
    custom = input("Customize DemandIndex & Rainfall? (y/n, default=n): ").strip().lower()
except Exception:
    custom = "n"
if custom == "y":
    try:
        demand_val = int(input(f"Enter DemandIndex (default {defaults.get('DemandIndex', 60)}): "))
        rainfall_val = float(input(f"Enter Rainfall mm (default {defaults.get('Rainfall', 5)}): "))
    except:
        print("Invalid input, using defaults.")

# Forecast - use last SEQ_LEN sequence from scaled data
last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)
next_pred = model.predict(last_seq, verbose=0)

# Build new row: predicted target values followed by zeros for the rest (if any),
# but since targets == features, we just take next_pred as full vector of length n_targets.
# However Keras Dense outputs len(targets) which equals len(features) here — good.
if next_pred.size == 0:
    print("Model did not produce a next-step prediction.")
    forecast_price = np.zeros(len(targets))
else:
    # If the model output size equals number of targets (== number of features), we can inverse directly.
    # Ensure next_pred is shaped (1, n_features) for inverse_transform
    pred_row = next_pred[0]
    if pred_row.shape[0] < N_FEATURES:
        # pad the remaining features (shouldn't happen here)
        pred_row_full = np.hstack([pred_row, np.array([rainfall_val, holiday_val, demand_val])])
        pred_row_full = np.hstack([pred_row_full, np.zeros(N_FEATURES - pred_row_full.shape[0])])
    else:
        pred_row_full = pred_row[:N_FEATURES]

    # For the special 3 extra fields (Rainfall, Holiday, DemandIndex) override with chosen manual values
    # Find their indices
    try:
        idx_rain = features.index("Rainfall")
        idx_hol = features.index("Holiday")
        idx_dem = features.index("DemandIndex")
        pred_row_full[idx_rain] = rainfall_val
        pred_row_full[idx_hol] = holiday_val
        pred_row_full[idx_dem] = demand_val
    except ValueError:
        # if any missing, ignore
        pass

    # Inverse transform: since scaler expects shape (n_samples, n_features)
    forecast_price = scaler.inverse_transform(pred_row_full.reshape(1, -1))[0][:len(targets)]

# Confidence interval (approx) using MAE per product if available
if preds_rescaled.size > 0:
    mae_per_product = np.mean(np.abs(y_test_rescaled - preds_rescaled), axis=0)
    if mae_per_product.shape[0] != len(targets):
        mae_per_product = np.resize(mae_per_product, len(targets))
else:
    mae_per_product = np.zeros(len(targets))

lower = forecast_price - mae_per_product
upper = forecast_price + mae_per_product
last_prices = df[targets].iloc[-1].values
scenario = ["Tataas" if f > l else "Bababa" for f, l in zip(forecast_price, last_prices)]

# Build forecast DataFrame
forecast_df = pd.DataFrame({
    "Produkto": targets,
    f"Presyo_{today.strftime('%b%d')}": [f"{p:.2f}" for p in last_prices],
    f"Forecast_{tomorrow.strftime('%b%d')}": [f"{p:.2f}" for p in forecast_price],
    "Lower80": [f"{p:.2f}" for p in lower],
    "Upper80": [f"{p:.2f}" for p in upper],
    "Scenario": scenario
})

# Save forecast to CSV & Excel
forecast_csv_path = f"outputs/AI_Forecast_{tomorrow.strftime('%Y%m%d')}.csv"
forecast_df.to_csv(forecast_csv_path, index=False)
print(f"Tomorrow forecast saved to {forecast_csv_path}")

excel_output = BytesIO()
with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
    forecast_df.to_excel(writer, index=False, sheet_name="Forecast")
excel_bytes = excel_output.getvalue()
excel_path = f"outputs/AI_Forecast_{tomorrow.strftime('%Y%m%d')}.xlsx"
with open(excel_path, "wb") as f:
    f.write(excel_bytes)
print(f"Tomorrow forecast saved to {excel_path}")

# Author
print("\nAuthor: BERNIESORIANO")
