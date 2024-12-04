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


class StockForm(forms.Form):
    company_with_tickers = forms.ChoiceField(
        choices=load_ticker_choices(),
        label="View Stocks and their ticker symbols",
        widget=forms.Select(attrs={'id': 'id_company_with_tickers', 'class': 'form-control'}),
        required=False
    )
    choices = forms.ChoiceField(
        label='Select one of the stocks from the list:',
        choices=[('', 'Select the Company Name')] + [('AAPL', 'Apple'), ('GOOGL', 'Google'), ('NVDA', 'NVIDIA'), ('TSLA', 'Tesla'), ('WBD', 'Warner Bros Discovery'), ('AMZN', 'Amazon'), ('INTC', 'Intel'), ('NFLX', 'Netflix'), ('META', 'Meta'), ('F', 'Ford Motor')],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'}),
        initial=''
    )
    search = forms.CharField(
        label='Enter the ticker symbol for the stock:',
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    choices1 = forms.ChoiceField(
        label='Select Historical or Prediction data:',
        choices=[(0,'Historical Data'),(1,'Prediction Data')],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'}),
        initial=''
    )
    choices2 = forms.ChoiceField(
        label='Select future or past data:',
        choices=[('1D','Daily'),('5D','Weekly'),('1M','Monthly'),('6M','Quarterly'),('YTD','Yearly'),('MAX','Max')],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'}),
        initial=''
    )
def update_choices():
    file_path = Path(r'C:\Users\sathw\Downloads\SE Project\project\stockproject\models')
    list_stocks = []
    for fname in os.listdir(file_path):
        list_stocks.append((str(fname[:-3]), ''))
    return list_stocks
