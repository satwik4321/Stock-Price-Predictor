import os
from django import forms
from django.conf import settings
from pathlib import Path

def load_ticker_choices():
    choices = [('','Select the Company Name')]  # Placeholder text
    file_path = os.path.join(settings.BASE_DIR, 'myapp/data/stocks.csv')
    try:
        with open(file_path, 'r') as file:
            next(file)  # Skip the header
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
        label="View Stocks and their ticker symbols",
        widget=forms.Select(attrs={'id': 'id_company_with_tickers', 'class': 'form-control'}),
        required=False
    )
<<<<<<< HEAD
    choices = forms.ChoiceField(label='Select one of the stocks from the list:',choices=[('Select One','None Selected'),('AAPL', 'Apple'), ('GOOGL', 'Google'),('NVDA','NVIDIA'),('TSLA','Tesla'),('WBD','Warner Bros Discovery'),('AMZN','Amazon'),('INTC','Intel'),('NFLX','Netflix'),('META','Meta'),('F','Ford Motor')],required=True)
    search = forms.CharField(label='Enter the ticker symbol for the stock:', max_length=100,required=False)
    timeframes = forms.ChoiceField(label='Select one of the stocks from the list:',choices=[('1 week') ],required=True)
    
import os
from pathlib import Path
=======
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

>>>>>>> 103a0814b8276dbc8a5286f089196f10b90d2a35
def update_choices():
    file_path = Path(r'C:\Users\sathw\Downloads\SE Project\project\stockproject\models')
    list_stocks = []
    for fname in os.listdir(file_path):
        list_stocks.append((str(fname[:-3]), ''))
    return list_stocks
