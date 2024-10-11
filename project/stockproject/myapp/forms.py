import os
from django import forms
from django.conf import settings

def load_ticker_choices():
    choices = [('','Select One')]
    file_path = os.path.join(settings.BASE_DIR, 'myapp/data/stocks.csv')
    try:
        with open(file_path, 'r') as file:
            next(file)  # Skip the header if there is one
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    symbol, name = parts[0], parts[1]
                    choices.append((symbol, f"{name}, {symbol}"))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return choices

class StockForm(forms.Form):
    company_with_tickers = forms.ChoiceField(
        choices=load_ticker_choices(),
        label="View Company Names and Tickers",
        required=False
    )
    choices = forms.ChoiceField(label='Select one of the stocks from the list:',choices=[('Select One','None Selected'),('AAPL', 'Apple'), ('GOOGL', 'Google'),('NVDA','NVIDIA'),('TSLA','Tesla'),('WBD','Warner Bros Discovery'),('AMZN','Amazon'),('INTC','Intel'),('NFLX','Netflix'),('META','Meta'),('F','Ford Motor')],required=False)
    search = forms.CharField(label='Enter the ticker symbol for the stock:', max_length=100,required=False)

import os
from pathlib import Path
def update_choices():
    file_path = Path(r'C:\Users\gogin\OneDrive\Documents\GitHub\Stock-Price-Predictor\project\stockproject\models')
    list_stocks=[]
    list_stocks.append(('Select One','None Selected'))
    for fname in os.listdir(file_path):
        list_stocks.append((str(fname[:-3]),''))
    return list_stocks

class StockForm(forms.Form):
    search = forms.CharField(label='Enter the ticker symbol for the stock:', max_length=100,required=False)
    choices=update_choices()
    print(choices)
    stock_choices=[('Select One','None Selected'),('AAPL', 'Apple'), ('GOOGL', 'Google'),('NVDA','NVIDIA'),('TSLA','Tesla'),('WBD','Warner Bros Discovery'),('AMZN','Amazon'),('INTC','Intel'),('NFLX','Netflix'),('META','Meta'),('F','Ford Motor')]
    choices = forms.ChoiceField(label='Select one of the stocks from the list:',choices=stock_choices,required=False)
    
