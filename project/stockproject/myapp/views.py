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
from keras.layers import Dense, LSTM, Dropout
import os #Importing OS to save a file to the machine
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from pathlib import Path
import os

import yfinance as yf
#from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
def train_model(name,data,input):
    data=data.filter(['Close'])
    data=data.values
    training_len=int(np.ceil(len(data)*0.75))
    test_len=len(data)-training_len
    #print(data)
    data=data.reshape(-1,1)
    #print(data)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    #print(scaled_data[0][0])
    #print(scaled_data)

    train_data = scaled_data[0:int(training_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    print(x_train[0])
    print(y_train[0])
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    file_path = Path(r'C:\Users\gogin\OneDrive\Documents\GitHub\Stock-Price-Predictor\project\stockproject\models')
    name_f=str(name+'.h5')
    full_path=os.path.join(file_path,name_f)
    # Build the LSTM model
    if input==0:
        file_path = Path(r'C:\Users\gogin\OneDrive\Documents\GitHub\Stock-Price-Predictor\project\stockproject\models')
        name_f=str(name+'.h5')
        full_path=os.path.join(file_path,name_f)
        full_path=Path(full_path)
        if full_path.is_file():
            model = keras.models.load_model(full_path)    
        else:
            model = Sequential()
            model.add(Dense(60))
            model.add(Dense(100))
            model.add(Dense(100))
            model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1))
        
    # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=1, epochs=1)
            file_path = Path(r'C:\Users\gogin\OneDrive\Documents\GitHub\Stock-Price-Predictor\project\stockproject\models')
            name_f=str(name+'.h5')
            full_path=os.path.join(file_path,name_f)
            model.save(full_path)
    else:
        model = keras.models.load_model(full_path)
    #test_data = scaled_data[training_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = data[training_len:, :]
    for i in range(training_len,len(data)):
        x_test.append(data[i-60:i, 0])
    # Convert the data to a numpy array
    x_test = np.array(x_test) 
    # Evaluate the model on test data
    test_loss = model.evaluate(x_test, y_test)
    print('Test Loss:', test_loss)
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
            data_stock=stock.history(start="2020-01-01",end=None)
            print(Name)
            print(data_stock)
            save_stock_data(Name, data_stock)
        else:
            print(form.errors)
        
        train_model(Name,data_stock,input)
        
        data={}
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
            csv_filepath = os.path.join(r'C:\Users\gogin\OneDrive\Documents\GitHub\Stock-Price-Predictor\project\stockproject\myapp\data', csv_filename)

            data_stock.to_csv(csv_filepath)

            return f"Stock data saved to: {csv_filename}"
        
        except Exception as e:
            return f"Error: {str(e)}"


def home(request):
    form = StockForm()
    return render(request, 'myapp/home.html', {'form': form})
