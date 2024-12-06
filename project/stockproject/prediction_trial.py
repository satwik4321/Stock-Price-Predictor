import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Download Stock Data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date)
    return stock_data

# Step 2: Preprocess Data
def preprocess_data_lstm(stock_data):
    # Extract closing prices and reshape
    close_prices = stock_data['Close'].values.reshape(-1, 1)

    # Scale data to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    return scaled_data, scaler

# Step 3: Create Sequences for LSTM
def create_sequences(data, sequence_length):
    X = []
    y = []

    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])  # Sequence of prices
        y.append(data[i, 0])  # Next price

    return np.array(X), np.array(y)

# Step 4: Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout for regularization
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Final output layer for price prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5: Visualization
def visualize_predictions(actual_prices, predicted_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, label="Actual Prices")
    plt.plot(predicted_prices, label="Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Main workflow
if __name__ == "__main__":
    # Define parameters
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    sequence_length = 60  # Number of days to look back for each prediction

    # Step 1: Download stock data
    stock_data = download_stock_data(ticker, start_date, end_date)

    # Step 2: Preprocess data
    scaled_data, scaler = preprocess_data_lstm(stock_data)

    # Step 3: Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

    # Split into training and test sets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Step 4: Build and train LSTM model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    # Step 5: Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))  # Rescale to original range
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Step 6: Visualize predictions
    visualize_predictions(actual_prices, predictions)
