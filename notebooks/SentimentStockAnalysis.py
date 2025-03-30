#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

