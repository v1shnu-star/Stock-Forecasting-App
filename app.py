import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model(r'C:\Users\vishn\Documents\Documents\ML Project\Stock_Market_Prediction_ML\Stock Predictions Model.keras')



st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2012-01-01'
end = '2024-05-07'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)
st.write(
    """
    **MA50 (50-day Moving Average)**: This line shows the average stock price over the last 50 days. The red line represents this 50-day moving average, and the green line represents the actual stock price.

    **Price vs MA50**: This plot helps to visualize how the current stock price compares with the average price over the past 50 days. A moving average smooths out short-term fluctuations and highlights longer-term trends.
    """
)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)
st.write(
    """
    **MA50 (50-day Moving Average)**: (red line) This line shows the average stock price over the last 50 days.
    
    **MA100 (100-day Moving Average)**:  (blue line) This line shows the average stock price over the last 100 days.
    
    **Price vs MA50 vs MA100**: This plot compares the actual stock price with two moving averages: the 50-day and 100-day moving averages. This helps in understanding both short-term and long-term trends.
    """
)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)
st.write(
    """
    **MA100 (100-day Moving Average)**: (red line) This line shows the average stock price over the last 100 days.
    
    **MA200 (200-day Moving Average)**: (blue line) This line shows the average stock price over the last 200 days.
    
    **Price vs MA100 vs MA200**: This plot compares the actual stock price with two moving averages: the 100-day and 200-day moving averages. This provides a view of longer-term trends and how the price fluctuates in comparison.
    """
)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
st.write(
    """
    **Predicted Price**: (red line) This line represents the prices predicted by the model.
    
    **Original Price**: (green line) This line represents the actual stock prices.
    
    **Original Price vs Predicted Price**: This plot compares the model's predicted prices with the actual stock prices to visualize how well the model performs.
    """
)