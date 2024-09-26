from django import forms
import os
from pathlib import Path
def update_choices():
    file_path = Path(r'C:\Users\sathw\Downloads\SE Project\project\stockproject\models')
    list_stocks=[]
    list_stocks.append(('Select One','None Selected'))
    for fname in os.listdir(file_path):
        list_stocks.append((str(fname[:-3]),''))
    return list_stocks

class StockForm(forms.Form):
    search = forms.CharField(label='Enter the ticker symbol for the stock:', max_length=100,required=False)
    choices=update_choices()
    print(choices)
    choices = forms.ChoiceField(label='Select one of the stoks from the list:',choices=[('Select One','None Selected'),('AAPL', 'Apple'), ('GOOGL', 'Google'),('NVDA','NVIDIA'),('TSLA','Tesla'),('WBD','Warner Bros Discovery'),('AMZN','Amazon'),('INTC','Intel'),('NFLX','Netflix'),('META','Meta'),('F','Ford Motor')],required=False)
