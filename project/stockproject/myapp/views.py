from django.shortcuts import render
from keras.callbacks import EarlyStopping, ModelCheckpoint
from .forms import StockForm
from django.http import JsonResponse
import pandas as pd
import numpy as np
from .forms import StockForm
import matplotlib.pyplot as plt
import seaborn as sns
from django.http import HttpRequest
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout, TimeDistributed
from keras.optimizers import Adam
import random
import time
import os #Importing OS to save a file to the machine
sns.set_style('whitegrid')

plt.style.use("fivethirtyeight")
from pathlib import Path
import os

import yfinance as yf
#from pandas_datareader import data as pdr

from sklearn.preprocessing import MinMaxScaler
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import os

from django.shortcuts import render
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.embed import components
import yfinance as yf
import pandas as pd
from datetime import datetime

from datetime import date, timedelta
from django.shortcuts import render, redirect
from django.http import JsonResponse


def fetch_stock_data(request):
    global stocks
    # Example: Fetch data for all stocks
    # Use yfinance or any other library to fetch stock data
    data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        data[stock] = ticker.info.get("longName", "Unknown Company")
    
    return JsonResponse(data)
base_path = os.getenv("PROJECT_ROOT", ".")
data_path = os.path.join(base_path, "data", "input.txt")
base_path1=base_path
def create_lstm_data_train(data, time_steps):
    x, y = [], []
    training_len=int(np.ceil(len(data)*0.50))
    data=data[:training_len]


    if len(data) - (3*time_steps)<1:
        time=int(len(data)/3)-1
        time_steps=int(len(data)/3)-1
    else:
        time=len(data) - (2*time_steps)
    for i in range(time):
        x.append(data[i:(i + time_steps), 0])    
        y.append(data[i + time_steps:i+(2*time_steps), 0])

    return np.array(x), np.array(y)
def create_multi_step_sequences(data, sequence_length, n_future_steps):
    """
    Create sequences for multi-step prediction.
    
    :param data: Scaled time series data.
    :param sequence_length: Number of timesteps in input sequence.
    :param n_future_steps: Number of future timesteps to predict.
    :return: Features (X) and targets (y) for multi-step prediction.
    """
    X = []
    y = []
    sequence_length=int(sequence_length)
    n_future_steps=int(n_future_steps)
    for i in range(sequence_length, len(data) - n_future_steps + 1):
        X.append(data[i - sequence_length:i, 0])  # Sequence of input data
        y.append(data[i:i + n_future_steps, 0])   # Corresponding future steps

    return np.array(X), np.array(y)
def build_multi_step_lstm(input_shape, n_future_steps):
    """
    Build an LSTM model for multi-step prediction.
    
    :param input_shape: Shape of the input data (sequence_length, 1).
    :param n_future_steps: Number of future steps to predict.
    :return: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=n_future_steps))  # Output future steps
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
def predict_future(model, initial_sequence, n_future_steps, scaler):
    """
    Predict multiple future steps based on the current sequence.

    :param model: Trained LSTM model.
    :param initial_sequence: Last sequence of data to start predictions.
    :param n_future_steps: Number of future steps to predict.
    :param scaler: Scaler object to inverse transform predictions.
    :return: Array of predicted values.
    """
    predictions = []
    current_sequence = initial_sequence.copy()  # Start with the last known sequence

    for _ in range(n_future_steps):
        # Predict the next time step
        prediction = model.predict(current_sequence.reshape(1, current_sequence.shape[0], 1))
        predictions.append(prediction[0, 0])

        # Update the current sequence (remove the oldest value and append the prediction)
        current_sequence = np.append(current_sequence[1:], prediction)

    # Rescale predictions back to the original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions
def create_lstm_data_test(data, time_steps):
    x, y = [], []
    training_len=int(np.ceil(len(data)*0.25))
    if type(time_steps)==str:
        time_steps=training_len
    data=data[training_len:len(data)-(2*time_steps)]
    if len(data) - (3*time_steps)<1:
        time=int(len(data)/3)-1
        time_steps=int(len(data)/3)-1

    else:
        time=len(data) - (2*time_steps)
    
    for i in range(time):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps:i+(2*time_steps), 0])
    return np.array(x), np.array(y)
def preprocess_data_lstm(stock_data):
    # Extract closing prices and reshape
    close_prices = stock_data['Open'].values.reshape(-1, 1)

    # Scale data to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    return scaled_data, scaler
def create_sequences(data, sequence_length):
    X = []
    y = []
    sequence_length=int(sequence_length)
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])  # Sequence of prices
        y.append(data[i, 0])  # Next price

    return np.array(X), np.array(y)
def train_model(name,X,y,input,scaler,size,choice,stock_data):
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    size1=0
    if size=="MAX":
        size1="MAX"
        size=int(len(X_train)/3)
    request = HttpRequest()
    request.method = 'GET'    
    if int(choice)==0:
        if size1=="MAX":
            script,div=home(request,0,name,size1)
        else:
            script,div=home(request,0,name,int(size))
    else:
            size=int(size)
            for epoch in range(1, 11):
                # Simulate training with sleep
                time.sleep(1)  # Replace with model training code
                progress = epoch * 10  # Simulate progress in %
                loss = random.uniform(1, 3)  # Simulate loss value
            #x,y=create_lstm_data_train(data,size)
            #x = x.reshape((x.shape[0], x.shape[1], 1))
            #y = y.reshape((y.shape[0], y.shape[1], 1))  # Ensure target matches the output shape

            name_f=str(name+'.h5')
            full_path=os.path.join(data_path,name_f)
            if input==0:
                name_f=str(name+'.h5')
                full_path=os.path.join(data_path,name_f)
                full_path=Path(full_path)
                early_stopping = EarlyStopping(
                monitor='val_loss',       # Metric to monitor (e.g., validation loss)
                patience=10,              # Number of epochs with no improvement before stopping
                restore_best_weights=True # Restore the best weights at the end of training
                )

                if full_path.is_file():
                    model = keras.models.load_model(full_path)    
                else:
                    input_shape=(X_train.shape[1],1)
                    model = Sequential()
                    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
                    model.add(Dropout(0.2))  # Dropout for regularization
                    model.add(LSTM(units=50, return_sequences=False))
                    model.add(Dropout(0.2))
                    model.add(Dense(units=25))
                    model.add(Dense(units=1))
                    
                    # Compile the model
                    optimizer=Adam(learning_rate=0.018)
                    model.compile(optimizer=optimizer, loss='mean_absolute_error')
                    model=build_multi_step_lstm((X_train.shape[1], 1), 360)

                    # Create a fake request object
                    request = HttpRequest()
                    # Optionally, you can set request.method or request.path
                    request.method = 'GET'
                    model.fit(X_train, y_train, batch_size=128, epochs=10,callbacks=[early_stopping])
                    name_f=str(name+'.h5')
                    full_path=os.path.join(base_path,'models',name_f)
                    model.save(full_path)
            else:
                name_f=str(name+'.h5')
                full_path=os.path.join(base_path1,"models",name_f)
                model = keras.models.load_model(full_path)
            test_loss = model.evaluate(X_test, y_test)
            print('Test Loss:', test_loss)
            predictions = model.predict(X_test)
            
            
            actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
            y=y.reshape(-1,1)
            request = HttpRequest()
            request.method = 'GET'
            # Initialize the form
            form = StockForm()
            future_prices=pd.DataFrame(predictions)
            interval = "1m"  # 1-minute interval

            next_days = get_next_n_days(size) #Get the next n dates
            scaler.inverse_transform(y_test.reshape(-1, 1))
            predictions=model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            predictions=predictions.reshape(-1)
            MAE=0
    
            for i in range(len(predictions)):
                MAE+=abs(predictions[i]-actual_prices[i])
            
            MAE/=len(predictions)
            print("Mean absolute error:",MAE)
            predictions=predictions[:size]
            print("-----------------------",stock_data[-1])
            print(predictions[0])
            if predictions[0]<stock_data[-1]:
                predictions=predictions+(stock_data[-1]-predictions[0])
            stock_data=pd.DataFrame({"Date":next_days,"Open":predictions})
            # Reset index and flatten MultiIndex columns
            stock_data.reset_index(inplace=True)
            stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns]

            # Ensure the Date column exists and is timezone-free
            if 'Date' in stock_data.columns:
                stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
            else:
                raise ValueError("Date column is missing in the fetched stock data.")
            # Dynamically calculate candlestick width based on interval
            interval_mapping = {
                "1m": 60 * 1000,        # 1 minute in milliseconds
                "5m": 5 * 60 * 1000,    # 5 minutes in milliseconds
                "15m": 15 * 60 * 1000,  # 15 minutes in milliseconds
                "30m": 30 * 60 * 1000,  # 30 minutes in milliseconds
                "1h": 60 * 60 * 1000,   # 1 hour in milliseconds
                "1d": 24 * 60 * 60 * 1000,  # 1 day in milliseconds
            }
            width = interval_mapping.get(interval, 12 * 60 * 60 * 1000)  # Default to 12 hours if interval is unknown
            stock_data["Previous_Open"] = stock_data["Open"].shift(1)  # Shift "Open" by 1
            stock_data["Greater_Than_Previous"] = stock_data["Open"] > stock_data["Previous_Open"]

            # Get indices where condition is True
            inc = stock_data[stock_data["Greater_Than_Previous"]].index.tolist()

            stock_data["Previous_Open"] = stock_data["Open"].shift(1)  # Shift "Open" by 1
            stock_data["Greater_Than_Previous"] = stock_data["Open"] < stock_data["Previous_Open"]
            stock_data["Equal"] = stock_data["Open"] == stock_data["Previous_Open"]
            dec = stock_data[stock_data["Greater_Than_Previous"]].index.tolist()
            same = stock_data[stock_data["Equal"]].index.tolist()
            # Create a candlestick chart
            stock_data["Previous_Open"] = stock_data["Open"].shift(1)
            stock_data['Previous_Open_plus_one']=stock_data["Equal"]+10
            source_inc = ColumnDataSource(data=stock_data.iloc[inc])
            source_dec = ColumnDataSource(data=stock_data.iloc[dec])
            source_same= ColumnDataSource(data=stock_data.iloc[same])
            p = figure(
                x_axis_type="datetime",
                height=600,
                width=100,
                title="Candlestick Chart",
                sizing_mode="stretch_width"
            )
            p.grid.grid_line_alpha = 0.3

            
            # Plot increasing candles (green)
            p.segment(x0='Date', y0='Previous_Open', x1='Date', y1='Open', color="green", source=source_inc)
            # Plot decreasing candles (red)
            p.segment(x0='Date', y0='Previous_Open', x1='Date', y1='Open', color="red", source=source_dec)
            p.segment(
    x0="Date", x1="Date",  # Use the same Date for start and end (vertical line)
    y0="Previous_Open", y1="Open", # Same values for Open and Close (horizontal line)
    color="blue", line_width=2, source=source_same)
            # Add hover tool for interactivity
            hover = HoverTool(
                tooltips=[
                    ("Date", "@Date{%F}"),
                    ("Open", "@Open{0.2f}")
                ],
                formatters={
                    '@Date': 'datetime',
                },
                mode='vline'
            )
            p.add_tools(hover)

            # Generate the script and div for the chart
            script, div = components(p)

            # Render the home.html template with the Bokeh chart
    
    return script,div
    #Don't delete the code below
    MAE=0
    
    for i in range(len(y_pred)):
        MAE+=abs(y_pred[i]-y_actual[i])
    
    MAE/=len(y_pred)
    print("Mean absolute error:",MAE)

def collect_history(request):
    if request.method == 'POST':
        form=StockForm(request.POST)
        if form.is_valid():
            choice=form.cleaned_data['choices1']
            timeframe=form.cleaned_data.get("choices2")
            # Initialize variables to ensure they are available in all code paths
            company_info = form.cleaned_data.get('company_with_tickers')
            input=0
            if company_info=='':
                company_info=form.cleaned_data.get('choices')
                input=1
            if ',' in company_info:
                company_name, ticker = company_info.split(', ')
                company_name = company_name.strip()  # Clean up any leading/trailing whitespace
                ticker = ticker.strip()  # Clean up any leading/trailing whitespace
            else:
                ticker = company_info.strip()
                company_name = yf.Ticker(ticker)
                company_name=company_name.info["longName"]
            # Log the selected company and ticker for debugging
            print(f"Selected company: {company_name}, ticker: {ticker}")
            start_date = "2017-01-03"
            stock_data = yf.download(ticker, start=start_date)
            timeframe1=365
            #print(data_stock)
            date=str(stock_data.index[0])
            if start_date[:10]!=date[:10]:
                timeframe1=60
            save_stock_data(ticker, stock_data)
        
            data_close=stock_data['Close'].values.reshape(-1,1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            close_prices_scaled = scaler.fit_transform(data_close)
            scaled_data, scaler = preprocess_data_lstm(stock_data)
            # Step 3: Create sequences
            sequence_length=60

            # Split into training and test sets
            
            
            X,y=create_multi_step_sequences(scaled_data,360,360)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            split_ratio = 0.8
            split_index = int(len(X) * split_ratio)
            script, div=train_model(ticker,X,y,input,scaler,timeframe,choice,stock_data['Open'].values.reshape(-1, 1))
            # Prepare the plot title
            if choice=="0":
                plot_title = f"{company_name} ({ticker}) Historical Data"
            else:
                plot_title = f"{company_name} ({ticker}) Predicted Trend"

            return render(request, 'myapp/home.html', {'form': form, 'script': script, 'div': div, 'plot_title': plot_title, 'data': stock_data.to_dict()})
        else:
            print("Form errors:", form.errors)
    return JsonResponse({"error": "Invalid request"}, status=400)


#Store the data collected from the Collect_History function to be saved to a csv file

def save_stock_data(stock_name, stock_data):
        try:
            stock = yf.Ticker(stock_name)
            data_stock = stock.history(start = "2017-01-01", end = None)
            if data_stock.empty:
                    return JsonResponse({"error": "No data found for stock symbol"}, status = 4040)
            csv_filename= f"{stock_name}_stock_data.csv"
            csv_filepath = os.path.join(base_path,'data',csv_filename)
            data_stock.to_csv(csv_filepath)
            return f"Stock data saved to: {csv_filename}"
        except Exception as e:
            return f"Error: {str(e)}"



# Function to get the next n dates
def get_next_n_days(n):
    today = date.today()
    return [(today + timedelta(days=i)) for i in range(n)]




def home(request,call=None,name=None,size=None):
    # Initialize the form
    if name==None:
        name="AAPL"

    form = StockForm()
    if size!=None and size!="MAX" and size<367:
        start_date = (datetime.today() - timedelta(days=int(size))).strftime('%Y-%m-%d')
        print("------------------------------",start_date)
        stock_data = yf.download(
        name,
        start=start_date,
        progress=False
    )
    
    else:
        # Fetch stock data using yfinance
        stock_data = yf.download(
            name,
            progress=False
        )
    
    # Reset index and flatten MultiIndex columns
    stock_data.reset_index(inplace=True)
    stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns]

    # Ensure the Date column exists and is timezone-free
    if 'Date' in stock_data.columns:
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
    else:
        raise ValueError("Date column is missing in the fetched stock data.")

    # Rename remaining columns for standardization
    stock_data.rename(columns={
        'Open_AAPL': 'Open',
        'High_AAPL': 'High',
        'Low_AAPL': 'Low',
        'Close_AAPL': 'Close',
        'Adj Close_AAPL': 'Adj_Close',
        'Volume_AAPL': 'Volume',
    }, inplace=True)

    # Dynamically calculate candlestick width based on interval
    interval_mapping = {
        "1m": 60 * 1000,        # 1 minute in milliseconds
        "5m": 5 * 60 * 1000,    # 5 minutes in milliseconds
        "15m": 15 * 60 * 1000,  # 15 minutes in milliseconds
        "30m": 30 * 60 * 1000,  # 30 minutes in milliseconds
        "1h": 60 * 60 * 1000,   # 1 hour in milliseconds
        "1d": 24 * 60 * 60 * 1000,  # 1 day in milliseconds
    }
    interval="1m"
    width = interval_mapping.get(interval, 12 * 60 * 60 * 1000)  # Default to 12 hours if interval is unknown

    # Create a candlestick chart
    inc = stock_data['Close'] > stock_data['Open']
    dec = stock_data['Open'] > stock_data['Close']

    source_inc = ColumnDataSource(data=stock_data[inc])
    source_dec = ColumnDataSource(data=stock_data[dec])

    p = figure(
        x_axis_type="datetime",
        height=600,
        width=1000,
        sizing_mode="stretch_width"
    )
    p.grid.grid_line_alpha = 0.3

    # Plot increasing candles (green)
    p.segment(x0='Date', y0='Low', x1='Date', y1='High', color="green", source=source_inc)
    #p.vbar(x='Date', width=width, top='Open', bottom='Close', fill_color="#D5E1DD", line_color="black", source=source_inc)

    # Plot decreasing candles (red)
    p.segment(x0='Date', y0='Low', x1='Date', y1='High', color="red",source=source_dec)
    #p.vbar(x='Date', width=width, top='Open', bottom='Close', fill_color="#F2583E", line_color="black", source=source_dec)

    # Add hover tool for interactivity
    hover = HoverTool(
        tooltips=[
            ("Datetime", "@Date{%F %T}"),
            ("Open", "@Open{0.2f}"),
            ("High", "@High{0.2f}"),
            ("Low", "@Low{0.2f}"),
            ("Close", "@Close{0.2f}"),
        ],
        formatters={
            '@Datetime': 'datetime',
        },
        mode='vline'
    )
    p.add_tools(hover)
    # Generate the script and div for the chart
    script, div = components(p)
    if call==0:
        return script, div
    # Render the home.html template with the Bokeh chart
    return render(request, 'myapp/home.html', {'form': form, 'script': script, 'div': div})