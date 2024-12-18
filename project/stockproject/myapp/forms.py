import os
from django import forms
from django.conf import settings
from pathlib import Path

def load_ticker_choices():
    choices = [('','Select the Company Name')]
    file_path = os.path.join(settings.BASE_DIR, 'myapp/data/stocks.csv')
    try:
        with open(file_path, 'r') as file:
            next(file)  # Skip the header
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    symbol, name = parts[0].strip(), parts[1].strip()
                    choices.append((f"{symbol}, {name}", f"{symbol}, {name}"))
    except Exception as e:
        print(f"An error occurred: {e}")
    #print("Choices loaded:", choices) 
    return choices
global list_stocks
list_stocks=[('', 'Select the Company Name')] + [('AAPL', 'Apple Inc.'), ('GOOGL', 'Alphabet Inc'), ('NVDA', 'NVIDIA Corporation'), ('TSLA', 'Tesla Inc.'), ('WBD', 'Warner Bros. Discovery Inc. Series A'), ('AMZN', 'Amazon.com Inc.'), ('INTC', 'Intel Corporation'), ('NFLX', 'Netflix Inc.'), ('META', 'Meta Platforms Inc.'), ('F', 'Ford Motor Company')]

class StockForm(forms.Form):

    company_with_tickers = forms.ChoiceField(
        choices=load_ticker_choices(),
        label="View Stocks and their ticker symbols",
        widget=forms.Select(attrs={'id': 'id_company_with_tickers', 'class': 'form-control'}),
        required=False
    )

    choices = forms.ChoiceField(
        label='Select one of the stocks from the list:',
        choices=list_stocks,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'}),
        initial=''
    )
    
    choices1 = forms.ChoiceField(
        label='Select Historical or Prediction data:',
        choices=[('0','Historical Data'),('1','Prediction Data')],
        required=True,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    choices2 = forms.ChoiceField(
        label='Select future or past data:',
        choices=[(2,'Daily'),(8,'Weekly'),(31,'Monthly'),(91,'Quarterly'),(366,'Yearly'),('MAX','Max')],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'}),
        initial=''
    )

    def update_choices():
        base_path = os.getenv("PROJECT_ROOT", ".")
        file_path = os.path.join(base_path,"models")
        list_stocks = []
        for fname in os.listdir(file_path):
            list_stocks.append((str(fname[:-3]), ''))
        #return list_stocks
