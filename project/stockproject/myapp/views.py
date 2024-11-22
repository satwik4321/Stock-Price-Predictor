from django.shortcuts import render
from .forms import StockForm
from django.http import JsonResponse
import pandas as pd
import numpy as np
from .forms import StockForm
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout, TimeDistributed
from keras.optimizers import Adam
from django.shortcuts import render, redirect
import os #Importing OS to save a file to the machine
sns.set_style('whitegrid')

plt.style.use("fivethirtyeight")
from pathlib import Path
import os

import yfinance as yf
#from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
def create_lstm_data_train(data, time_steps):
 x, y = [], []
 training_len=int(np.ceil(len(data)*0.50))
 data=data[:training_len]
 for i in range(len(data) - (2*time_steps)):
    x.append(data[i:(i + time_steps), 0])
    y.append(data[i + time_steps:i+(2*time_steps), 0])
 return np.array(x), np.array(y)

def create_lstm_data_test(data, time_steps):
 x, y = [], []
 training_len=int(np.ceil(len(data)*0.25))
 data=data[training_len:len(data)-(2*time_steps)]
 for i in range(len(data) - (2*time_steps)):
    x.append(data[i:(i + time_steps), 0])
    y.append(data[i + time_steps:i+(2*time_steps), 0])
 return np.array(x), np.array(y)

def train_model(name,data,input,scaler,size):
    x,y=create_lstm_data_train(data,size)
    file_path = Path(r'C:\Users\gogin\OneDrive\Documents\GitHub\SE Project\project\stockproject\models')
    name_f=str(name+'.h5')
    full_path=os.path.join(file_path,name_f)


    if input==0:
        file_path = Path(r'C:\Users\gogin\OneDrive\Documents\GitHub\SE Project\project\stockproject\models')
        name_f=str(name+'.h5')
        full_path=os.path.join(file_path,name_f)
        full_path=Path(full_path)
<<<<<<< HEAD
=======
        early_stopping = EarlyStopping(
        monitor='val_loss',       # Metric to monitor (e.g., validation loss)
        patience=10,              # Number of epochs with no improvement before stopping
        restore_best_weights=True # Restore the best weights at the end of training
        )

        '''model_checkpoint = ModelCheckpoint(
        filepath=full_path,       # Path to save the best model
        monitor='val_loss',       # Metric to monitor
        save_best_only=True,      # Save only when the metric improves
        verbose=1)                 # Print a message when the model is saved'''

>>>>>>> eef20da5d44c95ebb0c41f6b72e9955c7caef270
        if full_path.is_file():
            model = keras.models.load_model(full_path)    
        else:
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape=(len(x[0]), 1)))
            input_shape=(len(x[0]),1)
            model.add(LSTM(64, return_sequences=True))
            model.add(LSTM(32,return_sequences=True))
            model.add(TimeDistributed(Dense(1)))
            model.add(LSTM(128, return_sequences=True, input_shape=(len(x[0]), 1)))
            input_shape=(len(x[0]),1)
            model.add(LSTM(64, return_sequences=True))
            model.add(LSTM(32,return_sequences=True))
            model.add(TimeDistributed(Dense(1)))
        
            # Compile the model
            optimizer=Adam(learning_rate=0.018)
            model.compile(optimizer=optimizer, loss='mean_absolute_error')
<<<<<<< HEAD
            model.fit(x, y, batch_size=128, epochs=200)
            file_path = Path(r'C:\Users\gogin\OneDrive\Documents\GitHub\SE Project\project\stockproject\models')
=======
            

            # Create a fake request object
            request = HttpRequest()
            # Optionally, you can set request.method or request.path
            request.method = 'GET'
            my_view(request)
            model.fit(x, y, batch_size=128, epochs=400,callbacks=[early_stopping])
            file_path = Path(r'C:\Users\sathw\Downloads\SE Project\project\stockproject\models')
>>>>>>> eef20da5d44c95ebb0c41f6b72e9955c7caef270
            name_f=str(name+'.h5')
            full_path=os.path.join(file_path,name_f)
            print(full_path)
            model.save(full_path)
    else:
        model = keras.models.load_model(full_path)

    x,y=create_lstm_data_test(data,size)
    test_loss = model.evaluate(x, y)

    x,y=create_lstm_data_test(data,size)
    test_loss = model.evaluate(x, y)
    print('Test Loss:', test_loss)
    y_pred=model.predict(x[0].reshape(1,len(x[0]),1))
    y_pred=y_pred.reshape(-1,1)
    y=y.reshape(-1,1)
    y_pred = scaler.inverse_transform(y_pred)
    y_actual = scaler.inverse_transform(y)
    y_actual=y_actual[:len(y_pred)]
    request = HttpRequest()
    request.method = 'GET'
    home(request)
    MAE=0
    for i in range(len(y_pred)):
        MAE+=abs(y_pred[i]-y_actual[i])
    MAE/=len(y_pred)
    print("Mean absolute error:",MAE)

    y_pred=model.predict(x[0].reshape(1,len(x[0]),1))
    y_pred=y_pred.reshape(-1,1)
    y=y.reshape(-1,1)
    y_pred = scaler.inverse_transform(y_pred)
    y_actual = scaler.inverse_transform(y)
    y_actual=y_actual[:len(y_pred)]
    MAE=0
    for i in range(len(y_pred)):
        MAE+=abs(y_pred[i]-y_actual[i])
    MAE/=len(y_pred)
    print("Mean absolute error:",MAE)

def collect_history(request):
    if request.method == 'POST':
        form=StockForm(request.POST)
        print(request.POST)
        input=0
        if form.is_valid():
            if form.cleaned_data['search']!="":
                Name=form.cleaned_data['search']
            else:
                if form.cleaned_data['choices']!="Select One":
                    Name=form.cleaned_data['choices']
                    input=1
                    
            stock=yf.Ticker(Name)
            print(stock)
            start_date = "2017-01-03"
            csv_filename= f"{Name}_stock_data.csv"
            csv_filepath = os.path.join(r'C:\Users\sathw\Downloads\SE Project\project\stockproject\myapp\data', csv_filename)
            data_stock = yf.download('AAPL', start=start_date)
            timeframe=365

            date=str(data_stock.index[0])
            
            print("date:",date[:10],start_date[:10])
            
            if start_date[:10]!=date[:10]:
                timeframe=60
            print("stock_name:",stock.info['symbol'])
            print("timeframe:",timeframe)
            save_stock_data(Name, data_stock)
        else:
            print(form.errors)
        data_close=data_stock['Close'].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices_scaled = scaler.fit_transform(data_close)
        train_model(Name,close_prices_scaled,input,scaler,timeframe)
        # Return a JSON response
        return render(request, 'myapp/home.html', {'form': form, 'data': data_stock.to_dict()})
    return JsonResponse({"error": "Invalid request"}, status=400)


#Store the data collected from the Collect_History function to be saved to a csv file

def save_stock_data(stock_name, stock_data):

        try:
            stock = yf.Ticker(stock_name)
            data_stock = stock.history(start = "2017-01-01", end = None)

            if data_stock.empty:
                    return JsonResponse({"error": "No data found for stock symbol"}, status = 4040)
            
            project_root = os.path.dirname(os.path.abspath(__file__))
            stock_data_folder = os.path.join(project_root, 'stockproject')
            
            csv_filename= f"{stock_name}_stock_data.csv"
            csv_filepath = os.path.join(r'C:\Users\gogin\OneDrive\Documents\GitHub\SE Project\project\stockproject\myapp\data', csv_filename)
            data_stock.to_csv(csv_filepath)

            return f"Stock data saved to: {csv_filename}"
        
        except Exception as e:
            return f"Error: {str(e)}"

def handle_stock_submission(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            company_with_tickers = form.cleaned_data.get('company_with_tickers')
            choices = form.cleaned_data.get('choices')
            search = form.cleaned_data.get('search')

            # Check that only one of the options is filled
            if (company_with_tickers or choices) and not search:
                # Process form using company_with_tickers or choices
                pass
            elif search and not (company_with_tickers or choices):
                # Process form using search
                pass
            else:
                # Handle error case or re-prompt user
                pass

            return redirect('some_success_url')
        else:
            # Handle the form errors
            return render(request, 'your_template.html', {'form': form})
    else:
        form = StockForm()
        return render(request, 'your_template.html', {'form': form})




from django.shortcuts import render
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.embed import components
import yfinance as yf
import pandas as pd
from datetime import datetime


def home(request):
    # Initialize the form
    form = StockForm()

    interval = "1m"  # 1-minute interval
    start_date = "2024-10-24"
    end_date = "2024-10-25"

    # Fetch stock data using yfinance
    stock_data = yf.download(
        "AAPL",
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False
    )
    print(stock_data)
    # Reset index and flatten MultiIndex columns
    stock_data.reset_index(inplace=True)
    stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns]

    # Dynamically rename Datetime/Date column
    if 'Datetime_' in stock_data.columns:
        stock_data.rename(columns={'Datetime_': 'Date'}, inplace=True)
    elif 'Date_' in stock_data.columns:
        stock_data.rename(columns={'Date_': 'Date'}, inplace=True)

    # Ensure the Date column exists and is timezone-free
    if 'Date' in stock_data.columns:
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
    '''else:
        raise ValueError("Date column is missing in the fetched stock data.")'''

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
        title="Candlestick Chart",
        sizing_mode="stretch_width"
    )
    p.grid.grid_line_alpha = 0.3

    # Plot increasing candles (green)
    p.segment(x0='Datetime', y0='Low', x1='Datetime', y1='High', color="black", source=source_inc)
    p.vbar(x='Datetime', width=width, top='Open', bottom='Close', fill_color="#D5E1DD", line_color="black", source=source_inc)

    # Plot decreasing candles (red)
    p.segment(x0='Datetime', y0='Low', x1='Datetime', y1='High', color="black", source=source_dec)
    p.vbar(x='Datetime', width=width, top='Open', bottom='Close', fill_color="#F2583E", line_color="black", source=source_dec)

    # Add hover tool for interactivity
    hover = HoverTool(
        tooltips=[
            ("Datetime", "@Datetime{%F %T}"),
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

    # Render the home.html template with the Bokeh chart
    return render(request, 'myapp/home.html', {'form': form, 'script': script, 'div': div})

def my_view(request):
    context = {'message': 'Hello from Django!'}
    return render(request, 'templates\myapp\home.html', context)
