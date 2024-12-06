
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Download data from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Preprocess data
def preprocess_data(stock_data):
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data = stock_data.dropna()
    stock_data['Future_Close'] = stock_data['Close'].shift(-1)
    stock_data = stock_data.dropna()  # Drop the last row as it doesn't have future data
    return stock_data

# Step 3: Train-test split
def prepare_features(stock_data):
    X = stock_data[['Close', 'Volume', 'Return']].values  # Features
    y = stock_data['Future_Close'].values  # Target
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a simple Linear Regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Step 5: Predict and evaluate
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return predictions

# Step 6: Visualization
def visualize_predictions(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Prices", alpha=0.7)
    plt.plot(predictions, label="Predicted Prices", alpha=0.7)
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    plt.show()

# Main workflow
if __name__ == "__main__":
    ticker = "AAPL"  # Example: Apple Inc.
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    # Step 1: Download data
    stock_data = download_stock_data(ticker, start_date, end_date)

    # Step 2: Preprocess data
    stock_data = preprocess_data(stock_data)

    # Step 3: Prepare features and split the data
    X_train, X_test, y_train, y_test = prepare_features(stock_data)

    # Step 4: Train the model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate the model
    predictions = evaluate_model(model, X_test, y_test)

    # Step 6: Visualize results
    visualize_predictions(y_test, predictions)
