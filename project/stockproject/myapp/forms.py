from django import forms

class StockForm(forms.Form):
    search = forms.CharField(label='Enter the ticker symbol for the stock:', max_length=100,required=False)
    choices = forms.ChoiceField(label='Select one of the stoks from the list:',choices=[('Select One','None Selected'),('AAPL', 'Apple'), ('GOOGL', 'Google'),('NVDA','NVIDIA'),('TSLA','Tesla'),('WBD','Warner Bros Discovery'),('AMZN','Amazon'),('INTC','Intel'),('NFLX','Netflix'),('META','Meta'),('F','Ford Motor')],required=False)
