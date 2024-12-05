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

def train_model(name,data,input,scaler,size,choice):
    print("size---------------",size)
    #print("input at beginning:------------",data)
    request = HttpRequest()
    request.method = 'GET'
    if int(choice)==0:
        script,div=home(request,0,name,size)
    else:
            size=int(size)
            for epoch in range(1, 11):
                # Simulate training with sleep
                time.sleep(1)  # Replace with model training code
                progress = epoch * 10  # Simulate progress in %
                loss = random.uniform(1, 3)  # Simulate loss value
            x,y=create_lstm_data_train(data,size)
            x = x.reshape((x.shape[0], x.shape[1], 1))
            y = y.reshape((y.shape[0], y.shape[1], 1))  # Ensure target matches the output shape

            name_f=str(name+'.h5')
            full_path=os.path.join(data_path,name_f)
            #print("x----------------------",x)
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
                    model = Sequential()
                    model.add(LSTM(128, return_sequences=True, input_shape=(len(x[0]), 1)))
                    model.add(LSTM(64, return_sequences=True))
                    model.add(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
                    model.add(TimeDistributed(Dense(1)))
                    
                    # Compile the model
                    optimizer=Adam(learning_rate=0.018)
                    model.compile(optimizer=optimizer, loss='mean_absolute_error')
                    

                    # Create a fake request object
                    request = HttpRequest()
                    # Optionally, you can set request.method or request.path
                    request.method = 'GET'
                    model.fit(x, y, batch_size=128, epochs=10,callbacks=[early_stopping])
                    name_f=str(name+'.h5')
                    full_path=os.path.join(base_path,'models',name_f)
                    #print(full_path)
                    model.save(full_path)
            else:
                name_f=str(name+'.h5')
                full_path=os.path.join(base_path1,"models",name_f)
                model = keras.models.load_model(full_path)
            print("size--------------",size)
            x,y=create_lstm_data_test(data,size)
            print(len(x[0]))
            test_loss = model.evaluate(x, y)
            print('Test Loss:', test_loss)
            #print("x-----------------------",x[0].reshape(1,len(x[0]),1))
            #y_pred=model.predict(x[0].reshape(1,len(x[0]),1))
            #y_pred=model.predict(x)
            y_pred = model.predict(x[-2].reshape(1, len(x[-1]), 1))
            y_pred=y_pred.reshape(-1,1)
            print("y_pred:---------------",y_pred)
            y=y.reshape(-1,1)
            #print("x actual-------------",x[-1])
            y_actual=y[:len(y_pred)]
            request = HttpRequest()
            request.method = 'GET'
            # Initialize the form
            form = StockForm()
            future_prices=pd.DataFrame(y_pred)
            interval = "1m"  # 1-minute interval

            next_days = get_next_n_days(len(y_pred)) #Get the next n dates
            y_pred=y_pred.reshape(-1)
            print("len of y_pred:",len(y_pred))
            stock_data=pd.DataFrame({"Date":next_days,"Open":y_pred})
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
            print("indices:",dec,inc,same)
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
            #p.vbar(x='Date', width=width, top='Open', bottom='Close', fill_color="#D5E1DD", line_color="black", source=source_inc)
            # Plot decreasing candles (red)
            p.segment(x0='Date', y0='Previous_Open', x1='Date', y1='Previous_Open_plus_one', color="red", source=source_dec)
            #p.vbar(x='Date', width=width, top='Open', bottom='Close', fill_color="#F2583E", line_color="black", source=source_dec)
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
    '''MAE=0
    
    for i in range(len(y_pred)):
        MAE+=abs(y_pred[i]-y_actual[i])
    
    MAE/=len(y_pred)
    print("Mean absolute error:",MAE)'''

def collect_history(request):
    if request.method == 'POST':
        form=StockForm(request.POST)
        if form.is_valid():
            choice=form.cleaned_data['choices1']
            timeframe=form.cleaned_data.get("choices2")
            print("required_data_type:",choice)
            # Initialize variables to ensure they are available in all code paths
            company_info = form.cleaned_data.get('company_with_tickers')
            input=0
            print("Company Info:", company_info)  # If a choice from  the ticker symbol
            if company_info=='':
                print("none selected")
                company_info=form.cleaned_data.get('choices')
                input=1
            print("Company Info1:", company_info) 
            if ',' in company_info:
                print("company info:",company_info)
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
            data_stock = yf.download(ticker, start=start_date)
            print("data_loaded")
            timeframe1=365
            #print(data_stock)
            date=str(data_stock.index[0])
            if start_date[:10]!=date[:10]:
                timeframe1=60
            save_stock_data(ticker, data_stock)
        
            data_close=data_stock['Close'].values.reshape(-1,1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            close_prices_scaled = scaler.fit_transform(data_close)
            script, div=train_model(ticker,data_close,input,scaler,timeframe,choice)
            # Prepare the plot title
            if choice=="0":
                plot_title = f"{company_name} ({ticker}) Historical Data"
            else:
                plot_title = f"{company_name} ({ticker}) Predicted Trend"

            return render(request, 'myapp/home.html', {'form': form, 'script': script, 'div': div, 'plot_title': plot_title, 'data': data_stock.to_dict()})
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

    interval = "1m"  # 1-minute interval
    start_date = "2024-01-24"
    end_date = "2024-10-25"

    # Fetch stock data using yfinance
    stock_data = yf.download(
        name,
        start=start_date,
        end=end_date,
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
    p.segment(x0='Date', y0='Low', x1='Date', y1='High', color="red", line_dash="dashed",source=source_dec)
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