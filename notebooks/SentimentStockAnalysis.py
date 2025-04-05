#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os, re

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

#Naive Bayes Train Model Section
def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower().strip()

df_VADER['Clean Tweet'] = df_VADER['Tweet'].apply(clean_tweet)
X = df_VADER['Clean Tweet']
y = df_VADER['Sentiment Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(CountVectorizer(ngram_range=(1,2)), MultinomialNB())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)
st.subheader("Naive Bayes Model Evaluation Report")
st.dataframe(report_df, use_container_width=True)

# Test Section, the Naive Bayes needs improvement.
st.subheader("Test Custom Sentiment")
user_input = st.text_area("Enter a tweet or comment (Ctrl + Enter to submit):")

if user_input:
    cleaned_input = clean_tweet(user_input)
    prediction = model.predict([cleaned_input])[0]
    vader_score = analyzer.polarity_scores(user_input)['compound']

    if vader_score > 0.05:
        vader_sentiment = "Positive"
    elif vader_score < -0.05:
        vader_sentiment = "Negative"
    else:
        vader_sentiment = "Neutral"

    st.markdown("### Results")
    st.write(f"**Naive Bayes Sentiment:** {prediction}")
    st.write(f"**VADER Sentiment:** {vader_sentiment}")
    st.write(f"**VADER Sentiment Score:** {vader_score:.3f}")
