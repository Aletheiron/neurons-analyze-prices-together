import numpy as np
import yfinance as yf
import pandas as pd

nvidia_data=yf.Ticker('NVDA')
nvidia_data=nvidia_data.history(period='max')

# print(nvidia_data.head())
# print(nvidia_data.tail())

#del nvidia_data['Dividends']
#del nvidia_data['Stock Splits']

nvidia_data['Tomorrow']=nvidia_data['Close'].shift(-1)
nvidia_data['Target']=(nvidia_data['Tomorrow']>nvidia_data['Close']).astype(int)
nvidia_data['Price -1 day']=nvidia_data['Close'].shift(1)
nvidia_data['Difference']=(nvidia_data['Close']/nvidia_data['Price -1 day'])
nvidia_data=nvidia_data.copy(deep=True)
nvidia_data=nvidia_data[['Difference','Target']]
nvidia_data['Difference-1']=nvidia_data['Difference'].shift(1)
nvidia_data['Difference-2']=nvidia_data['Difference'].shift(2)
nvidia_data['Difference-3']=nvidia_data['Difference'].shift(3)
nvidia_data['Difference-4']=nvidia_data['Difference'].shift(4)
nvidia_data['Difference-5']=nvidia_data['Difference'].shift(5)
nvidia_data['Difference-6']=nvidia_data['Difference'].shift(6)
nvidia_data['Difference-7']=nvidia_data['Difference'].shift(7)

nvidia_data=nvidia_data.copy()
nvidia_data=nvidia_data[['Difference','Difference-1','Difference-2','Difference-3','Difference-4','Difference-5','Difference-6','Difference-7','Target']]
nvidia_data=nvidia_data.dropna()
nvidia_data=nvidia_data.copy(deep=True)

nvidia_data.to_csv('nvda_data',header=False, index=False)

# print(nvidia_data.head())
# print(nvidia_data.tail())