#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

st.set_page_config(page_title="Stock Price Visualization")

st.title("Stock Closing Prices Over Time")

file_path = os.path.join("..", "dataFiles", "stock_yfinance_data.csv")

df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"]).dt.date
df = df.sort_values(by="Date")

st.subheader("Data Information")
buffer = st.expander("Show Data Info")
with buffer:
    df_info = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.count()
    })
    df_info['Type'] = [str(dtype) for dtype in df.dtypes]
    st.dataframe(df_info)

st.subheader("Data Preview")
st.dataframe(df.head())

st.subheader("Stock Closing Prices")
fig, ax = plt.subplots(figsize=(10, 6))

for stock in df["Stock Name"].unique():
    stock_df = df[df["Stock Name"] == stock]
    ax.plot(stock_df["Date"], stock_df["Close"], label=stock)

ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.set_title("Stock Closing Prices Over Time")
ax.legend(loc="upper left")

st.pyplot(fig)

st.subheader("Stock Summary Statistics")
st.dataframe(df.groupby("Stock Name")["Close"].describe())


#VADER Sentiment Analysis Section
file_path_VADER = os.path.join("..", "dataFiles", "stock_tweets.csv")
df_VADER = pd.read_csv(file_path_VADER)
df_VADER['Date'] = pd.to_datetime(df_VADER['Date'])

st.subheader("VADER Sentiment Classification")
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']

if 'Tweet' in df_VADER.columns:
    df_VADER['Sentiment'] = df_VADER['Tweet'].apply(lambda tweet: analyze_sentiment(str(tweet)))

    st.subheader("Sentiment Analysis of Stock Tweets")
    df_VADER['Sentiment Label'] = df_VADER['Sentiment'].apply(lambda score: 'Positive' if score > 0.05 else ('Negative' if score < -0.05 else 'Neutral'))
    st.dataframe(df_VADER[['Date', 'Stock Name', 'Tweet', 'Sentiment', 'Sentiment Label']], use_container_width=True)

    st.subheader("Sentiment Distribution by Stock")
    sentiment_counts = df_VADER.groupby(['Stock Name', 'Sentiment Label']).size().unstack(fill_value=0)

    st.dataframe(sentiment_counts)

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Tweet Sentiment by Stock")
    st.pyplot(fig)

