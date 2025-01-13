import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller  # For stationarity check
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # Autocorrelation
from sklearn.preprocessing import MinMaxScaler  # Scaling for LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error



# Check if the time series is stationary
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The series is stationary")
    else:
        print("The series is not stationary")
        
# Plot autocorrelation and partial autocorrelation
def plot_acf_pacf(series):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(series, lags=50, ax=plt.gca())
    plt.subplot(122)
    plot_pacf(series, lags=50, ax=plt.gca())
    plt.tight_layout()
    plt.show()

# Sliding window transformation for time series
def create_supervised_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# Build the LSTM model for time series prediction
def build_lstm_model(X_train, X_val, y_train, y_val, epochs=2, batch_size=32):
    # Reshape the data for LSTM (samples, timesteps, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_lstm = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], 1), return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_lstm, y_val), verbose=2)
    
    # Make predictions on validation data
    predictions = model.predict(X_val_lstm)
    
    # Evaluate the model using RMSE
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f'Validation RMSE (LSTM): {rmse}')

    return model, predictions

# Prediction function for LSTM with downsampling
def lstm_prediction():
    train_file = 'C:/Users/Naim/rossmann-store-sales/data/train.csv'
    store_file = 'C:/Users/Naim/rossmann-store-sales/data/store.csv'

    # Load and merge data
    train_merged, _ = merge_data(train_file, None, store_file)

    # Convert 'Date' column to datetime and downsample sales data to weekly to reduce memory usage
    train_merged['Date'] = pd.to_datetime(train_merged['Date'])
    sales_series = train_merged.resample('W', on='Date')['Sales'].sum()  # Weekly aggregation

    # Check if the 'Sales' column exists
    if 'Sales' not in train_merged.columns:
        raise KeyError("'Sales' column is missing in the merged training data.")
    
    # Check stationarity
    check_stationarity(sales_series)

    # Plot autocorrelation and partial autocorrelation
    plot_acf_pacf(sales_series)

    # Create supervised learning data using sliding window
    window_size = 30  # Window size of 30 weeks
    X, y = create_supervised_data(sales_series.values, window_size)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data to (-1, 1) range
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Build and train the LSTM model
    model, predictions = build_lstm_model(X_train, X_val, y_train, y_val)

    return model, predictions