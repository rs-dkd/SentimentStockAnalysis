#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta, datetime
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
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


# Parameter grid for tuning
param_grid = {
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidfvectorizer__max_df': [0.8, 0.9],
    'multinomialnb__alpha': [0.1, 0.5, 1.0]
}

model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())

# GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

with st.spinner("Optimizing Naive Bayes model with GridSearchCV..."):
    grid_search.fit(X_train, y_train)

# Best parameters and score
st.write(f"Best parameters: {grid_search.best_params_}")
st.write(f"Best score: {grid_search.best_score_}")

grid_results_df = pd.DataFrame(grid_search.cv_results_)
grid_results_df = grid_results_df[['params', 'mean_test_score', 'std_test_score']].sort_values(by='mean_test_score', ascending=False)

# Show the results
st.subheader("GridSearchCV Results")
st.dataframe(grid_results_df, use_container_width=True)

grid_results_df['alpha_max_df'] = grid_results_df['params'].apply(
    lambda x: f"alpha={x['multinomialnb__alpha']}, max_df={x['tfidfvectorizer__max_df']}"
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=grid_results_df, x='alpha_max_df', y='mean_test_score', marker='o')

ax.set_title('GridSearchCV: Hyperparameter vs Mean Test Score')
ax.set_xlabel('Hyperparameter Combinations')
ax.set_ylabel('Mean Test Score')
ax.set_xticklabels(grid_results_df['alpha_max_df'], rotation=90, fontsize=8)
st.pyplot(fig)

# Generate classification report after fitting the best model
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)

# Display the model evaluation report
st.subheader("Optimized Naive Bayes Model Evaluation Report")
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

#Sentiment and Price Relation, could be better visualized
st.header("Relationship Between Sentiment and Price Movement")

stock_options = df["Stock Name"].unique()
selected_stock_temporal = st.selectbox("Select a stock for relationship analysis", stock_options)
stock_df = df[df["Stock Name"] == selected_stock_temporal].copy()
stock_tweets = df_VADER[df_VADER["Stock Name"] == selected_stock_temporal].copy()
stock_tweets['Date_Only'] = stock_tweets['Date'].dt.date
daily_sentiment = stock_tweets.groupby('Date_Only')['Sentiment'].agg(['mean', 'count']).reset_index()
daily_sentiment.columns = ['Date', 'Avg_Sentiment', 'Tweet_Count']
merged_df = pd.merge(stock_df, daily_sentiment, on='Date', how='left')
merged_df = merged_df.fillna({'Avg_Sentiment': 0, 'Tweet_Count': 0})
merged_df['Daily_Return'] = merged_df['Close'].pct_change() * 100
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price ($)', color=color)
ax1.plot(merged_df['Date'], merged_df['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Average Sentiment', color=color)
ax2.plot(merged_df['Date'], merged_df['Avg_Sentiment'], color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f'Stock Price and Twitter Sentiment for {selected_stock_temporal}')
fig.tight_layout()
st.pyplot(fig)

#Pattern Mining Section, up down ratio could be better represented
st.header("Tweet Pattern Mining")

def extract_patterns(tweets_df, price_df, threshold=1.0):
    tweets_df = tweets_df.copy()
    price_df = price_df.copy()
    if 'Daily_Return' not in price_df.columns:
        price_df['Daily_Return'] = price_df['Close'].pct_change() * 100
    
    tweets_df['Date_Only'] = pd.to_datetime(tweets_df['Date']).dt.date
    
    try:
        joined_df = tweets_df.merge(price_df[['Date', 'Daily_Return']], 
                                  left_on='Date_Only', 
                                  right_on='Date', 
                                  how='inner')
    except KeyError:
        st.error("Missing columns")
        return None, None
    
    if 'Clean Tweet' not in joined_df.columns:
        if 'Tweet' in joined_df.columns:
            joined_df['Clean Tweet'] = joined_df['Tweet'].apply(clean_tweet)
        else:
            st.error("Tweet content missing")
            return None, None
    
    significant_up = joined_df[joined_df['Daily_Return'] > threshold]
    significant_down = joined_df[joined_df['Daily_Return'] < -threshold]
    
    if len(significant_up) < 5 or len(significant_down) < 5:
        st.warning(f"Not enough data with movements above {threshold}% threshold")
        return None, None
    
    up_tweets = " ".join(significant_up['Clean Tweet'].dropna().astype(str))
    down_tweets = " ".join(significant_down['Clean Tweet'].dropna().astype(str))
    
    try:
        vectorizer = CountVectorizer(stop_words='english', min_df=2, max_features=100)
        if len(up_tweets) > 10 and len(down_tweets) > 10:
            up_down_matrix = vectorizer.fit_transform([up_tweets, down_tweets])
            terms = vectorizer.get_feature_names_out()
            
            up_freq = up_down_matrix[0].toarray()[0]
            down_freq = up_down_matrix[1].toarray()[0]
            
            terms_df = pd.DataFrame({
                'Term': terms,
                'Upward_Movement_Freq': up_freq,
                'Downward_Movement_Freq': down_freq
            })
            
            terms_df['Up_Down_Ratio'] = (terms_df['Upward_Movement_Freq'] + 0.1) / (terms_df['Downward_Movement_Freq'] + 0.1)
            
            upward_indicators = terms_df.sort_values('Up_Down_Ratio', ascending=False).head(10)
            
            downward_indicators = terms_df.sort_values('Up_Down_Ratio').head(10)
            
            return upward_indicators, downward_indicators
        else:
            st.warning("Not enough tweet content")
            return None, None
    except Exception as e:
        st.error(f"Error in extraction: {e}")
        return None, None

upward_indicators, downward_indicators = extract_patterns(
    stock_tweets, 
    stock_df,
    threshold=1.0
)

if upward_indicators is not None and downward_indicators is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Terms Associated with Upward Price Movement")
        st.dataframe(upward_indicators[['Term', 'Up_Down_Ratio']])
    
    with col2:
        st.subheader("Terms Associated with Downward Price Movement")
        st.dataframe(downward_indicators[['Term', 'Up_Down_Ratio']])
else:
    st.info("Could not extract patterns for this stock")

#Trading Strategy, needs some issues fixed for different thresholds/holding periods
st.header("Sentiment-Based Trading Strategy")

sentiment_threshold = st.slider("Sentiment Threshold", -1.0, 1.0, 0.2, 0.05)
holding_period = st.slider("Holding Period (Days)", 1, 10, 3)

def backtest_sentiment_strategy(merged_df, sentiment_threshold, holding_period):
    backtest_df = merged_df.copy()
    backtest_df = backtest_df.dropna(subset=['Avg_Sentiment', 'Close'])
    
    if len(backtest_df) < holding_period + 2:
        return None
    
    backtest_df['Position'] = 0
    backtest_df['Portfolio_Value'] = 0.0
    initial_investment = 10000
    current_cash = initial_investment
    current_shares = 0
    hold_counter = 0

    for i in range(len(backtest_df)):
        row = backtest_df.iloc[i]
        if hold_counter > 0:
            hold_counter -= 1
            if hold_counter == 0 and current_shares > 0:
                current_cash = current_shares * row['Close']
                current_shares = 0
                backtest_df.loc[backtest_df.index[i], 'Position'] = -1
        
        elif row['Avg_Sentiment'] > sentiment_threshold and current_shares == 0:
            shares_to_buy = current_cash / row['Close']
            current_shares = shares_to_buy
            current_cash = 0
            backtest_df.loc[backtest_df.index[i], 'Position'] = 1
            hold_counter = holding_period
        
        portfolio_value = current_cash + (current_shares * row['Close'])
        backtest_df.loc[backtest_df.index[i], 'Portfolio_Value'] = portfolio_value
    
    if current_shares > 0 and len(backtest_df) > 0:
        final_price = backtest_df.iloc[-1]['Close']
        current_cash = current_shares * final_price
        current_shares = 0
        backtest_df.loc[backtest_df.index[-1], 'Portfolio_Value'] = current_cash
    
    return backtest_df

backtest_results = backtest_sentiment_strategy(merged_df, sentiment_threshold, holding_period)

if backtest_results is not None and len(backtest_results) > 0:
    initial_investment = 10000
    initial_shares = initial_investment / backtest_results.iloc[0]['Close']
    buy_hold_values = backtest_results['Close'] * initial_shares
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(backtest_results['Date'], backtest_results['Portfolio_Value'], 
           label='Sentiment Strategy', color='blue')
    ax.plot(backtest_results['Date'], buy_hold_values, 
           label='Buy and Hold', color='gray', linestyle='--')
    buys = backtest_results[backtest_results['Position'] == 1]
    sells = backtest_results[backtest_results['Position'] == -1]
    
    if len(buys) > 0:
        ax.scatter(buys['Date'], buys['Portfolio_Value'], 
                 color='green', marker='^', s=100, label='Buy')
    
    if len(sells) > 0:
        ax.scatter(sells['Date'], sells['Portfolio_Value'], 
                 color='red', marker='v', s=100, label='Sell')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(f'Sentiment-Based Trading Strategy for {selected_stock_temporal}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    final_portfolio = backtest_results['Portfolio_Value'].iloc[-1]
    buy_hold_final = buy_hold_values.iloc[-1]
    
    total_return = ((final_portfolio - initial_investment) / initial_investment) * 100
    buy_hold_return = ((buy_hold_final - initial_investment) / initial_investment) * 100
    
    st.subheader("Strategy Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Initial Investment", f"${initial_investment:,.2f}")
    
    with col2:
        st.metric("Final Portfolio Value", f"${final_portfolio:,.2f}", 
                f"{total_return:.2f}%")
    
    with col3:
        st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%", 
                f"{total_return - buy_hold_return:.2f}%")
    num_buys = len(buys)
    num_sells = len(sells)
    
    st.write(f"Number of trades: {num_buys} buys, {num_sells} sells")
else:
    st.warning("Not enough data for backtesting")

#Stock Comparison, can add any other visualizations here
st.header("Stock Performance Comparison")

stocks_to_compare = st.multiselect("Select stocks to compare", 
                                 df["Stock Name"].unique(), 
                                 default=list(df["Stock Name"].unique())[:min(3, len(df["Stock Name"].unique()))])

if stocks_to_compare:
    comparison_df = df[df["Stock Name"].isin(stocks_to_compare)].copy()
    stock_metrics = []
    
    for stock in stocks_to_compare:
        stock_data = comparison_df[comparison_df["Stock Name"] == stock].copy()
        
        if len(stock_data) < 5:
            continue
        
        initial_price = stock_data["Close"].iloc[0]
        final_price = stock_data["Close"].iloc[-1]
        price_change = ((final_price - initial_price) / initial_price) * 100
        stock_data["Daily_Return"] = stock_data["Close"].pct_change()
        volatility = stock_data["Daily_Return"].std() * 100
        stock_tweets = df_VADER[df_VADER["Stock Name"] == stock]
        avg_sentiment = stock_tweets["Sentiment"].mean() if len(stock_tweets) > 0 else None
        sentiment_count = len(stock_tweets)
        
        stock_metrics.append({
            "Stock": stock,
            "Initial Price": initial_price,
            "Final Price": final_price,
            "Price Change (%)": price_change,
            "Volatility (%)": volatility,
            "Avg Tweet Sentiment": avg_sentiment if avg_sentiment is not None else 0,
            "Tweet Count": sentiment_count
        })
    metrics_df = pd.DataFrame(stock_metrics)
    st.subheader("Stock Performance Metrics")
    st.dataframe(metrics_df.set_index("Stock"))
    st.subheader("Performance Comparison")
    
    if len(metrics_df) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            metrics_df["Volatility (%)"], 
            metrics_df["Price Change (%)"],
            s=metrics_df["Tweet Count"] / 5 if metrics_df["Tweet Count"].max() > 0 else 100,
            c=metrics_df["Avg Tweet Sentiment"],
            cmap="coolwarm",
            alpha=0.7,
            vmin=-0.5, vmax=0.5
        )
        
        for i, stock in enumerate(metrics_df["Stock"]):
            ax.annotate(stock, 
                       (metrics_df["Volatility (%)"].iloc[i], metrics_df["Price Change (%)"].iloc[i]),
                       xytext=(5, 5), textcoords="offset points")
        
        ax.set_xlabel("Risk (Volatility %)")
        ax.set_ylabel("Return (Price Change %)")
        ax.set_title("Risk vs Return Comparison")
        ax.grid(alpha=0.3)
        
        plt.colorbar(scatter, label="Average Tweet Sentiment")
        st.pyplot(fig)
        st.subheader("Price Trends Comparison")
        
        price_trends = pd.DataFrame()
        
        for stock in stocks_to_compare:
            stock_data = comparison_df[comparison_df["Stock Name"] == stock].copy()
            if len(stock_data) > 0:
                stock_data = stock_data.sort_values('Date')
                stock_data['Normalized_Price'] = stock_data['Close'] / stock_data['Close'].iloc[0] * 100
                price_trends[stock] = stock_data.set_index('Date')['Normalized_Price']
        
        if not price_trends.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            price_trends.plot(ax=ax)
            ax.set_xlabel('Date')
            ax.set_ylabel('Normalized Price (Starting at 100)')
            ax.set_title('Normalized Price Comparison')
            ax.grid(alpha=0.3)
            ax.legend(title='Stock')
            st.pyplot(fig)
            
st.header("Stock Sentiment Network Analysis")
st.write("Explore relationships between stocks based on sentiment patterns.")

if 'Sentiment' not in df_VADER.columns:
    st.error("Sentiment data not found. Run VADER Sentiment Classification first.")
    st.stop()

df_VADER["Date_Only"] = df_VADER["Date"].dt.date
daily_sentiment = df_VADER.groupby(["Date_Only", "Stock Name"])["Sentiment"].mean().reset_index()

st.subheader("Select Time Range for Analysis")
col1, col2 = st.columns(2)

min_date = df_VADER["Date"].min().date()
max_date = df_VADER["Date"].max().date()
default_start = max(min_date, max_date - timedelta(days=60))

with col1:
    start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date, key="network_start_date")
with col2:
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="network_end_date")

if start_date == default_start:
    start_date = max(min_date, end_date - timedelta(days=60))

if start_date > end_date:
    st.error("Error: End date must be after start date.")
else:
    filtered_sentiment = daily_sentiment[
        (daily_sentiment["Date_Only"] >= start_date) & 
        (daily_sentiment["Date_Only"] <= end_date)
    ]
    
    if len(filtered_sentiment) == 0:
        st.warning("No sentiment data available for this date range.")
    else:
        sentiment_pivot = filtered_sentiment.pivot_table(
            index="Date_Only", columns="Stock Name", values="Sentiment", aggfunc="mean"
        ).fillna(0)
        
        valid_stocks = sentiment_pivot.columns[sentiment_pivot.count() > 2].tolist()
        
        if len(valid_stocks) < 2:
            st.warning("Not enough sentiment data. Select a wider date range.")
        else:
            sentiment_pivot = sentiment_pivot[valid_stocks]
            sentiment_corr = sentiment_pivot.corr()
            
            correlation_threshold = st.slider("Correlation Threshold", min_value=0.0, max_value=1.0, 
                                             value=0.2, step=0.05, key="corr_threshold",
                                             help="Show connections above this correlation value")
            
            st.subheader("Network Visualization Settings")
            col1, col2 = st.columns(2)
            with col1:
                network_layout = st.selectbox("Network Layout", 
                                            ["spring", "circular", "kamada_kawai", "spectral"],
                                            key="network_layout")
                node_size_factor = st.slider("Node Size Factor", 100, 1000, 500, 50, key="node_size")
            
            with col2:
                edge_weight_factor = st.slider("Edge Thickness", 1, 10, 3, 1, key="edge_weight")
                show_labels = st.checkbox("Show Stock Labels", value=True, key="show_labels")
            
            adj_matrix = sentiment_corr.copy()
            adj_matrix[adj_matrix < correlation_threshold] = 0
            np.fill_diagonal(adj_matrix.values, 0)
            
            G = nx.from_pandas_adjacency(adj_matrix)
            
            if len(G.edges()) == 0:
                st.warning(f"No connections found. Try lowering the correlation threshold.")
            else:
                sentiment_volume = filtered_sentiment.groupby("Stock Name").size()
                
                min_size = 10
                node_sizes = {}
                for stock in G.nodes():
                    if stock in sentiment_volume:
                        node_sizes[stock] = min_size + (sentiment_volume[stock] / sentiment_volume.max()) * node_size_factor
                    else:
                        node_sizes[stock] = min_size
                
                avg_sentiment = filtered_sentiment.groupby("Stock Name")["Sentiment"].mean()
                
                st.subheader("Stock Sentiment Network")
                fig, ax = plt.subplots(figsize=(10, 10))
                
                if network_layout == "spring":
                    pos = nx.spring_layout(G, seed=42)
                elif network_layout == "circular":
                    pos = nx.circular_layout(G)
                elif network_layout == "kamada_kawai":
                    pos = nx.kamada_kawai_layout(G)
                else:
                    pos = nx.spectral_layout(G)
                
                edge_weights = [adj_matrix.loc[u, v] * edge_weight_factor for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
                
                node_colors = []
                for node in G.nodes():
                    if node in avg_sentiment:
                        if avg_sentiment[node] > 0:
                            intensity = min(1.0, 0.3 + abs(avg_sentiment[node]))
                            node_colors.append((0, intensity, 0))
                        else:
                            intensity = min(1.0, 0.3 + abs(avg_sentiment[node]))
                            node_colors.append((intensity, 0, 0))
                    else:
                        node_colors.append((0.5, 0.5, 0.5))
                
                nx.draw_networkx_nodes(G, pos, 
                                      node_size=[node_sizes.get(node, min_size) for node in G.nodes()],
                                      node_color=node_colors, alpha=0.8)
                
                if show_labels:
                    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
                
                plt.axis('off')
                plt.title(f'Stock Sentiment Correlation Network ({start_date} to {end_date})')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("Network Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Stocks", f"{len(G.nodes())}")
                with col2:
                    st.metric("Number of Connections", f"{len(G.edges())}")
                with col3:
                    st.metric("Network Density", f"{nx.density(G):.3f}")
                
                st.subheader("Sentiment Correlation Heatmap")
                connected_stocks = []
                for i, stock1 in enumerate(valid_stocks):
                    for j, stock2 in enumerate(valid_stocks):
                        if i < j and abs(sentiment_corr.loc[stock1, stock2]) >= correlation_threshold:
                            if stock1 not in connected_stocks:
                                connected_stocks.append(stock1)
                            if stock2 not in connected_stocks:
                                connected_stocks.append(stock2)
                
                top_stocks = connected_stocks[:min(10, len(connected_stocks))] if len(connected_stocks) >= 3 else valid_stocks[:min(10, len(valid_stocks))]
                correlation_matrix = sentiment_corr.loc[top_stocks, top_stocks]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1,
                          linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
                plt.title('Sentiment Correlation Between Stocks')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("Sentiment Trend Comparison")
                st.write("Select two stocks to compare sentiment trends:")
                
                col1, col2 = st.columns(2)
                with col1:
                    stock1 = st.selectbox("First Stock", options=valid_stocks, index=0, key="stock1_select")
                with col2:
                    correlated_stocks = sentiment_corr[stock1].sort_values(ascending=False)
                    correlated_stocks = correlated_stocks[correlated_stocks.index != stock1]
                    suggested_idx = valid_stocks.index(correlated_stocks.index[0]) if len(correlated_stocks) > 0 else min(1, len(valid_stocks)-1)
                    stock2 = st.selectbox("Second Stock", options=valid_stocks, 
                                        index=min(suggested_idx, len(valid_stocks)-1), key="stock2_select")
                
                stock1_data = filtered_sentiment[filtered_sentiment["Stock Name"] == stock1]
                stock2_data = filtered_sentiment[filtered_sentiment["Stock Name"] == stock2]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(stock1_data["Date_Only"], stock1_data["Sentiment"], 
                      marker='o', linestyle='-', color='blue', label=stock1)
                ax.plot(stock2_data["Date_Only"], stock2_data["Sentiment"], 
                      marker='s', linestyle='-', color='red', label=stock2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Sentiment Score')
                ax.set_title(f'Sentiment Comparison: {stock1} vs {stock2}')
                ax.legend()
                ax.grid(alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                if len(stock1_data) > 3 and len(stock2_data) > 3:
                    comparison_df = pd.merge(
                        stock1_data[["Date_Only", "Sentiment"]],
                        stock2_data[["Date_Only", "Sentiment"]],
                        on="Date_Only", suffixes=("_1", "_2")
                    )
                    
                    if len(comparison_df) > 3:
                        correlation = comparison_df["Sentiment_1"].corr(comparison_df["Sentiment_2"])
                        st.metric("Sentiment Correlation", f"{correlation:.3f}")
                    else:
                        st.info("Not enough overlapping data to calculate correlation.")
                else:
                    st.info("Not enough data points to calculate correlation.")
                
                st.subheader("Price and Sentiment Integration")
                st.write("Visualize how sentiment trends align with price movements:")
                
                selected_stock = st.selectbox("Select Stock for Analysis", 
                                            options=valid_stocks, index=0, key="price_sentiment_stock")
                
                price_data = df[
                    (df["Stock Name"] == selected_stock) &
                    (df["Date"] >= start_date) &
                    (df["Date"] <= end_date)
                ].copy()
                
                sentiment_data = filtered_sentiment[filtered_sentiment["Stock Name"] == selected_stock].copy()
                
                if len(price_data) > 0 and len(sentiment_data) > 0:
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Price ($)', color='blue')
                    ax1.plot(price_data["Date"], price_data["Close"], color='blue', label='Price')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Sentiment', color='red')
                    ax2.plot(sentiment_data["Date_Only"], sentiment_data["Sentiment"], 
                           color='red', marker='o', linestyle='-', label='Sentiment')
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    
                    plt.title(f'{selected_stock}: Price vs. Sentiment')
                    fig.tight_layout()
                    
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                    
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    merged_data = pd.merge(
                        pd.DataFrame({'Date': price_data["Date"], 'Price': price_data["Close"]}),
                        sentiment_data[["Date_Only", "Sentiment"]],
                        left_on='Date', right_on='Date_Only', how='inner'
                    )
                    
                    if len(merged_data) > 3:
                        price_sentiment_corr = merged_data['Price'].corr(merged_data['Sentiment'])
                        st.metric("Price-Sentiment Correlation", f"{price_sentiment_corr:.3f}")
                    else:
                        st.info("Not enough overlapping data to calculate correlation.")
                else:
                    st.warning("Insufficient price or sentiment data for this stock.")