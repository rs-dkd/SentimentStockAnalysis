# Stock Price Movement Prediction Using Twitter Sentiment Analysis

## Table of Contents
1. [Overview](#overview)
2. [Team Members](#team-members)
3. [Synopsis](#synopsis)
4. [Problem Statement](#problem-statement)
5. [Dataset](#dataset)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Baseline Techniques](#baseline-techniques)
8. [Novelty/Innovation/Impact](#noveltyinnovationimpact)
9. [Code Structure](#code-structure)
10. [How to Run](#how-to-run)

---

## Overview
This project aims to predict short-term stock price movements by leveraging sentiment analysis of tweets related to the top 25 most watched stocks. By combining natural language processing (NLP) techniques, exploratory data analysis (EDA), and machine learning models, we aim to uncover actionable insights into how social media sentiment correlates with stock price fluctuations.

---

## Team Members
- **Reggie Segovia**
- **Eric Truong**
- **Jeffrey Smith**
- **Justin Sui**

---

## Synopsis
Social media platforms like Twitter play a pivotal role in shaping public perception and influencing market activity. This project seeks to quantify the relationship between collective sentiment expressed in tweets and subsequent stock price changes. 

Our approach involves:
- Conducting Exploratory Data Analysis (EDA) to understand trends and patterns.
- Cleaning and preprocessing tweet data using NLP techniques.
- Classifying stock-related sentiment using advanced algorithms.
- Building predictive models to forecast stock price movements based on sentiment analysis.

The ultimate goal is to provide investors with valuable, data-driven insights that demonstrate how social media perception can serve as an early indicator of market trends.

---

## Problem Statement
Can we accurately predict whether a stock's price will rise or fall the following day by combining Twitter sentiment analysis with historical stock price data?

---

## Dataset
We will use the ["Stock Tweets for Sentiment Analysis and Prediction"](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction) dataset from Kaggle. This dataset includes:
- Historical tweets related to the top 25 most watched stocks.
- Corresponding stock price data from Yahoo Finance (spanning 09/30/2021 to 09/30/2022).

Key files:
- `stock_yfinance_data.csv`: Contains historical stock price data.
- `stock_tweets.csv`: Contains tweets relating to the top 25 stocks at a respective time.

If necessary, we will supplement this dataset with live data from the Yahoo Finance API to ensure accuracy and relevance.

---

## Evaluation Metrics
To assess the performance of our models, we will use the following metrics:

### Classification Metrics
- **Accuracy**: Overall correctness of predictions.
- **Precision, Recall, F1-Score**: To evaluate the balance between false positives and false negatives.
- **ROC-AUC Score**: For evaluating the trade-off between true positive rate and false positive rate.

### Regression Metrics (if applicable)
- **Mean Absolute Error (MAE)**: To measure prediction errors in stock price regression tasks.

### Visualization Tools
- **Confusion Matrix**: To visualize classification results.

---

## Baseline Techniques
### Sentiment Classification
- **VADER**: A lexicon-based sentiment analysis tool tailored for social media text.
- **Custom-trained Naive Bayes classifiers**: To label tweets as positive, negative, or neutral.

### Text Processing
- Tokenization
- Stopword removal
- TF-IDF vectorization

### Classification Models
- Logistic Regression
- Random Forests
- Gradient Boosting

### Clustering
- K-Means or DBSCAN to group tweets into sentiment clusters and identify patterns.

### Pattern Mining
- Analyze frequent keyword patterns correlated with stock price movements.

---

## Novelty/Innovation/Impact
This project introduces a hybrid predictive model that integrates social media sentiment signals with historical stock price data. Key innovations include:
- **Temporal Alignment**: Experimenting with techniques to synchronize tweet timestamps with stock price movements.
- **Hybrid Approach**: Combining sentiment analysis and technical indicators for enhanced predictive power.
- **Sentiment-Enhanced Trading Signals**: Creating actionable insights for traders and financial analysts.

By bridging the gap between social media sentiment and market behavior, this project has the potential to revolutionize how investors leverage data to make informed decisions.

---

## Code Structure
```
project-root/
│
├── dataFiles/ # Folder containing all CSV files and raw datasets
│ ├── stock_yfinance_data.csv
│ ├── stock_tweets.csv
│ └── additional_data.csv
│
├── notebooks/ # Jupyter notebooks and corresponding python files (Streamlit) for EDA, modeling, and visualization
│ ├── SentimentStockAnalysis.py (where main code is located)
| └── SentimentStockAnalysis.ipynb
|
│
├── .gitignore # Specifies files and folders to exclude from version control
│
|
├── requirements.txt # Project dependencies
|
|
└── README.md # Project documentation (this file)
```

## How to Run

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

```bash
cd notebooks
streamlit run SentimentStockAnalysis.py
```
