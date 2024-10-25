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
    file_path = Path(r'project\stockproject\models')
    name_f=str(name+'.h5')
    full_path=os.path.join(file_path,name_f)

    if input==0:
        file_path = Path(r'project\stockproject\models')
        name_f=str(name+'.h5')
        full_path=os.path.join(file_path,name_f)
        full_path=Path(full_path)
        if full_path.is_file():
            model = keras.models.load_model(full_path)    
        else:
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape=(len(x[0]), 1)))
            input_shape=(len(x[0]),1)
            model.add(LSTM(64, return_sequences=True))
            model.add(LSTM(32,return_sequences=True))
            model.add(TimeDistributed(Dense(1)))
        
            # Compile the model
            optimizer=Adam(learning_rate=0.018)
            model.compile(optimizer=optimizer, loss='mean_absolute_error')
            model.fit(x, y, batch_size=128, epochs=200)
            file_path = Path(r'project\stockproject\models')
            name_f=str(name+'.h5')
            full_path=os.path.join(file_path,name_f)
            model.save(full_path)
    else:
        model = keras.models.load_model(full_path)

    x,y=create_lstm_data_test(data,size)
    test_loss = model.evaluate(x, y)
    print('Test Loss:', test_loss)
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
            start_date = "2017-01-01"
            data_stock = yf.download(stock.info['symbol'], start=start_date)
            print("stock_name:",stock.info['symbol'])
            save_stock_data(Name, data_stock)
        else:
            print(form.errors)
        data_close=data_stock['Close'].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices_scaled = scaler.fit_transform(data_close)
        train_model(Name,close_prices_scaled,input,scaler,10)
        # Return a JSON response
        return render(request, 'myapp/home.html', {'form': form, 'message': 'Data fetched successfully!', 'data': data_stock.to_dict()})
    return JsonResponse({"error": "Invalid request"}, status=400)


#Store the data collected from the Collect_History function to be saved to a csv file

def save_stock_data(stock_name, stock_data):

        try:
            stock = yf.Ticker(stock_name)
            data_stock = stock.history(start = "2020-01-01", end = None)

            if data_stock.empty:
                    return JsonResponse({"error": "No data found for stock symbol"}, status = 4040)
            
            project_root = os.path.dirname(os.path.abspath(__file__))
            stock_data_folder = os.path.join(project_root, 'stockproject')
            
            csv_filename= f"{stock_name}_stock_data.csv"
            csv_filepath = os.path.join(r'project\stockproject\myapp\data', csv_filename)

            data_stock.to_csv(csv_filepath)

            return f"Stock data saved to: {csv_filename}"
        
        except Exception as e:
            return f"Error: {str(e)}"


def home(request):
    form = StockForm()
    return render(request, 'myapp/home.html', {'form': form})
