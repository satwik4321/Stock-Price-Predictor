from django.shortcuts import render
from .forms import StockForm
from django.http import JsonResponse
import pandas as pd
import numpy as np
from .forms import StockForm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
#%matplotlib inline

# For reading stock data from yahoo
#from pandas_datareader.data import DataReader
import yfinance as yf
#from pandas_datareader import data as pdr
def run_python_function(request):
    if request.method == 'POST':
        form=StockForm(request.POST)
        print(request.POST)
        if form.is_valid():
            if form.cleaned_data['search']!="":
                Name=form.cleaned_data['search']
            else:
                if form.cleaned_data['choices']!="Select One":
                    Name=form.cleaned_data['choices']
            stock=yf.Ticker(Name)
            data_stock=stock.history(start="2020-01-01",end=None)
            print(Name)
            print(data_stock)
        else:
            print(form.errors)
        data={}
        # Return a JSON response
        return render(request, 'myapp/home.html', {'form': form, 'message': 'Data fetched successfully!', 'data': data_stock.to_dict()})
    return JsonResponse({"error": "Invalid request"}, status=400)

def home(request):
    form = StockForm()
    return render(request, 'myapp/home.html', {'form': form})