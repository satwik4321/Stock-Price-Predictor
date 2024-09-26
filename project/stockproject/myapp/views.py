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
from keras.layers import Dense, LSTM
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from pathlib import Path
import os
# Create a Path object
#%matplotlib inline

# For reading stock data from yahoo
#from pandas_datareader.data import DataReader
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
        #if i<= 61:
            #print(x_train)
            #print(y_train)
            #print()
        
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    file_path = Path(r'C:\Users\sathw\Downloads\SE Project\project\stockproject\models')
    name_f=str(name+'.h5')
    full_path=os.path.join(file_path,name_f)
    # Build the LSTM model
    if input==0:
        file_path = Path(r'C:\Users\sathw\Downloads\SE Project\project\stockproject\models')
        name_f=str(name+'.h5')
        full_path=os.path.join(file_path,name_f)
        full_path=Path(full_path)
        if full_path.is_file():
            #print("exists")
            model = keras.models.load_model(full_path)    
        else:
            #print(name)
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
        
    # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=1, epochs=1)
            file_path = Path(r'C:\Users\sathw\Downloads\SE Project\project\stockproject\models')
            name_f=str(name+'.h5')
            full_path=os.path.join(file_path,name_f)
            model.save(full_path)
    else:
        model = keras.models.load_model(full_path)
    test_data = scaled_data[training_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = data[training_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    print(predictions)
    print(len(predictions))
    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print("rmse: ",rmse)
    # Train the model

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
        else:
            print(form.errors)
        
        train_model(Name,data_stock,input)

        data={}
        # Return a JSON response
        return render(request, 'myapp/home.html', {'form': form, 'message': 'Data fetched successfully!', 'data': data_stock.to_dict()})
    return JsonResponse({"error": "Invalid request"}, status=400)

def home(request):
    form = StockForm()
    return render(request, 'myapp/home.html', {'form': form})