import streamlit as st 
import pandas as pd
import plotly
import plotly.express as px
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date, timedelta, datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit.components.v1 as components
import numpy as np
from transformers import pipeline
from openai import OpenAI
import plotly.graph_objects as go
import pytz
import os
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title='Trading Price Predictor', layout='wide')

# Define the base URLs for accessing the API
BASE_URLS = [
    # "https://api-handler-ddc-free-api.hf.space/v2"
    "https://api.ddc.xiolabs.xyz/v1"
]

# Initialize the OpenAI client with a specific base URL and API key
try:
    client = OpenAI(
        base_url=BASE_URLS[0],  # Using the first URL in the list
        # api_key="DDC-Free-For-Subscribers-YT-@DevsDoCode"
        api_key="Free-For-YT-Subscribers-@DevsDoCode-WatchFullVideo"
    )
except Exception as e:
    st.warning(f"Could not initialize OpenAI client: {str(e)}")
    # Create a dummy client that will be replaced with proper error handling in the functions
    class DummyClient:
        def __init__(self):
            self.chat = self
            self.completions = self
        
        def create(self, **kwargs):
            raise Exception("OpenAI API is not available")
    
    client = DummyClient()

# Load FinBERT Model and Tokenizer
@st.cache_resource
def load_finbert():
    try:
        tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        modeL = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        return tokenizer, modeL
    except Exception as e:
        st.warning(f"Could not load FinBERT model: {str(e)}")
        return None, None

tokenizer, modeL = load_finbert()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

# Sidebar inputs
stocks = (
    'AAPL',     # Apple
    'MSFT',     # Microsoft
    'GOOGL',    # Google
    'AMZN',     # Amazon
    'META',     # Meta (Facebook)
    'TSLA',     # Tesla
    'NVDA',     # NVIDIA
    'JPM',      # JPMorgan Chase
    'V',        # Visa
    'WMT',      # Walmart
    'JNJ',      # Johnson & Johnson
    'PG',       # Procter & Gamble
    'MA',       # Mastercard
    'HD',       # Home Depot
    'BAC',      # Bank of America
    'DIS',      # Disney
    'NFLX',     # Netflix
    'INTC',     # Intel
    'VZ',       # Verizon
    'KO',       # Coca-Cola
    'PEP',      # PepsiCo
    'ADBE',     # Adobe
    'CSCO',     # Cisco
    'NKE',      # Nike
    'MCD',      # McDonald's
    'SPY',      # S&P 500 ETF
    'QQQ',      # Nasdaq ETF
)
st.sidebar.header('Input Parameters')
# st.selectbox('Select dataset for prediction', stocks)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        # st.dataframe(df.tail(100000))
        # df
        if df.empty:
            st.error(f"No data found for {symbol}")
            return None
        return df
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def plot_chart(data, column, title):
    try:
        fig = px.line(data, x=data.index, y=column, title=title)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting chart: {str(e)}")

def tech_indicators(data):
    st.header("Visualize Technical Indicators")
    if data is None:
        st.error("No data available for technical analysis")
        return

    # Add date range selection
    st.write("Select a date range:")
    start_dat = st.date_input("Start Date", data.index.min().date())
    end_dat = st.date_input("End Date", data.index.max().date())
    
    if start_dat > end_dat:
        st.error("Start Date cannot be after End Date.")
        return
    
    # Filter data based on selected date range
    filtered_data = data[(data.index >= pd.Timestamp(start_dat)) & (data.index <= pd.Timestamp(end_dat))]
    
    # st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', 
                      ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA', 'Ichimoku'])

    try:
        # Ensure 'Close', 'High', 'Low' columns are squeezed into 1D
        close_series = filtered_data['Close'].squeeze() if filtered_data['Close'].ndim > 1 else filtered_data['Close']
        high_series = filtered_data['High'].squeeze() if filtered_data['High'].ndim > 1 else filtered_data['High']
        low_series = filtered_data['Low'].squeeze() if filtered_data['Low'].ndim > 1 else filtered_data['Low']

        def plot_close_price():
            fig = px.line(x=filtered_data.index, y=close_series, title='Closing Price')
            st.plotly_chart(fig)

        def plot_bollinger_bands():
            bb_indicator = BollingerBands(close_series, window=20, window_dev=2)
            bb = pd.DataFrame({
                'Close': close_series,
                'bb_h': bb_indicator.bollinger_hband(),
                'bb_l': bb_indicator.bollinger_lband()
            }, index=filtered_data.index)
            fig = px.line(bb, x=bb.index, y=['Close', 'bb_h', 'bb_l'], title='Bollinger Bands')
            st.plotly_chart(fig)

        def plot_ichimoku():
            ichimoku = IchimokuIndicator(high=high_series, low=low_series, window1=9, window2=26, window3=52)
            ichimoku_data = pd.DataFrame({
                'Close': close_series,
                'ichimoku_a': ichimoku.ichimoku_a(),
                'ichimoku_b': ichimoku.ichimoku_b()
            }, index=filtered_data.index)
            fig = px.line(ichimoku_data, x=ichimoku_data.index, y=['Close', 'ichimoku_a', 'ichimoku_b'], title='Ichimoku Cloud')
            st.plotly_chart(fig)

        # Plot the selected indicator
        if option == 'Close':
            plot_close_price()
        elif option == 'BB':
            plot_bollinger_bands()
        elif option == 'MACD':
            macd = MACD(close_series).macd()
            plot_chart(pd.DataFrame({'MACD': macd}, index=filtered_data.index).squeeze(), 'MACD', 'Moving Average Convergence Divergence')
        elif option == 'RSI':
            rsi = RSIIndicator(close_series).rsi()
            plot_chart(pd.DataFrame({'RSI': rsi}, index=filtered_data.index).squeeze(), 'RSI', 'Relative Strength Index')
        elif option == 'SMA':
            sma = SMAIndicator(close_series, window=14).sma_indicator()
            plot_chart(pd.DataFrame({'SMA': sma}, index=filtered_data.index).squeeze(), 'SMA', 'Simple Moving Average')
        elif option == 'EMA':
            ema = EMAIndicator(close_series, window=14).ema_indicator()
            plot_chart(pd.DataFrame({'EMA': ema}, index=filtered_data.index).squeeze(), 'EMA', 'Exponential Moving Average')
        elif option == 'Ichimoku':
            plot_ichimoku()

    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        st.error("Please make sure your data contains the required columns (Close, High, Low)")

def predict():
    # Add date range selection
    st.header("Predict Future Prices")
    st.write("Select a date range:")
    start_dat = st.date_input("Start Date", st.session_state.data.index.min().date())
    end_dat = st.date_input("End Date", st.session_state.data.index.max().date())
    
    if start_dat > end_dat:
        st.error("Start Date cannot be after End Date.")
        return
    
    # Filter data based on selected date range
    filtered_data = st.session_state.data[(st.session_state.data.index >= pd.Timestamp(start_dat)) & 
                                        (st.session_state.data.index <= pd.Timestamp(end_dat))]
    
    # User selects the model and number of days to forecast
    model_choice = st.radio('Choose a model', 
                            ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 
                             'KNeighborsRegressor', 'XGBoostRegressor'])
    num_days = st.number_input('How many days forecast?', value=5, min_value=1)
    num_days = int(num_days)
    
    if st.button('Predict'):
        # Choose and initialize the appropriate model based on user input
        if model_choice == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num_days, filtered_data)
        elif model_choice == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num_days, filtered_data)
        elif model_choice == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num_days, filtered_data)
        elif model_choice == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num_days, filtered_data)
        else:
            engine = XGBRegressor()
            model_engine(engine, num_days, filtered_data)
        

def model_engine(model, forecast_days, filtered_data):
    # Copy only the 'Close' column for modeling to avoid the chained assignment warning
    df = filtered_data[['Close']].copy()
    
    # Shift the 'Close' prices for forecasting
    df['Future'] = df['Close'].shift(-forecast_days)
    
    # Preparing data for scaling and training
    x_data = df[['Close']].values
    y_data = df['Future'].values
    
    # Apply scaler from session state
    x_scaled = st.session_state.scaler.fit_transform(x_data)
    
    # Preparing the forecast data, training, and testing data
    x_forecast = x_scaled[-forecast_days:]
    x_train = x_scaled[:-forecast_days]
    y_train = y_data[:-forecast_days]
    
    # Split the data
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
        x_train, y_train, test_size=0.2, random_state=7)
    
    # Train and evaluate the model
    model.fit(x_train_split, y_train_split)
    predictions = model.predict(x_test_split)
    
    # Calculate metrics
    r2 = r2_score(y_test_split, predictions)
    mae = mean_absolute_error(y_test_split, predictions)
    
    # MAPE calculation with handling for zero values
    non_zero_mask = y_test_split != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_test_split[non_zero_mask] - predictions[non_zero_mask]) / y_test_split[non_zero_mask])) * 100
    else:
        mape = np.nan  # If all values are zero, MAPE is undefined
    
    # Display evaluation metrics
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Forecasting future prices
    future_predictions = model.predict(x_forecast)
    
    st.subheader(f"{forecast_days}-Day Price Forecast")
    for day, price in enumerate(future_predictions, start=1):
        st.write(f"Day {day}: ${price:.2f}")


def tradingview():
    st.header("Live Chart")
    st.sidebar.empty()
    # Embed TradingView Widget
    tradingview_widget = """
    <div class="tradingview-widget-container">
    <div class="tradingview-widget-container__widget"></div>
    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
    {
    "width": "100%",
    "height": "700",
    "symbol": "NASDAQ:AAPL",
    "interval": "D",
    "timezone": "Etc/UTC",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "allow_symbol_change": true,
    "calendar": false,
    "support_host": "https://www.tradingview.com"
    }
    </script>
    </div>
    """

    # Render the TradingView widget in Streamlit
    components.html(tradingview_widget, height=700)

# Function to Predict Sentiment
def predict_sentiment(text):
    if tokenizer is None or modeL is None:
        # Fallback to a simple rule-based sentiment analysis
        positive_words = ["up", "rise", "gain", "profit", "bull", "growth", "positive", "increase", "higher", "surge"]
        negative_words = ["down", "fall", "loss", "bear", "decline", "negative", "decrease", "lower", "drop", "plunge"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "Positive"
            scores = {"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7}
        elif negative_count > positive_count:
            sentiment = "Negative"
            scores = {"Negative": 0.7, "Neutral": 0.2, "Positive": 0.1}
        else:
            sentiment = "Neutral"
            scores = {"Negative": 0.3, "Neutral": 0.4, "Positive": 0.3}

        return sentiment, scores
    
    # Use FinBERT if available
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = modeL(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    sentiments = ["Negative", "Neutral", "Positive"]
    sentiment_scores = {sentiments[i]: probs[0][i].item() for i in range(len(sentiments))}
    predicted_sentiment = sentiments[torch.argmax(probs)]
    return predicted_sentiment, sentiment_scores

# Function to Generate Context-Aware Explanation
def generate_explanation(headline, sentiment):
    try:
        # First try with OpenAI API
        prompt = f"""
        The following is a financial news headline: "{headline}"
        Sentiment analysis indicates that the sentiment is {sentiment}.
        
        Based on this information, provide an explanation of how this news might impact the stock or forex or crypto or futures&options market or the specific asset mentioned, provide a theoretical signal whether user should buy/sell or long/short the given asset. Consider historical market trends, potential investor reactions, and the language in the news.
        """
        
        try:
            response = client.chat.completions.create(
                model="provider-3/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial expert providing context-aware market analysis."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception:
            # Fallback analysis based on sentiment and keywords
            keywords = {
                "bullish": ["surge", "rise", "gain", "up", "higher", "positive", "growth", "profit"],
                "bearish": ["fall", "decline", "down", "lower", "negative", "loss", "drop", "plunge"],
                "volatility": ["volatile", "uncertain", "fluctuate", "swing", "volatility"],
                "trend": ["trend", "momentum", "direction", "pattern", "cycle"]
            }
            
            headline_lower = headline.lower()
            
            # Analyze keywords in the headline
            sentiment_indicators = {
                "bullish": sum(1 for word in keywords["bullish"] if word in headline_lower),
                "bearish": sum(1 for word in keywords["bearish"] if word in headline_lower),
                "volatility": sum(1 for word in keywords["volatility"] if word in headline_lower),
                "trend": sum(1 for word in keywords["trend"] if word in headline_lower)
            }
            
            # Generate a detailed explanation based on sentiment and keywords
            explanation = []
            
            # Add sentiment-based analysis
            if sentiment == "Positive":
                explanation.append("This positive news suggests a potential upward movement in the market.")
                if sentiment_indicators["bullish"] > 0:
                    explanation.append("The presence of bullish indicators strengthens this positive outlook.")
            elif sentiment == "Negative":
                explanation.append("This negative news suggests a potential downward movement in the market.")
                if sentiment_indicators["bearish"] > 0:
                    explanation.append("The presence of bearish indicators strengthens this negative outlook.")
            else:
                explanation.append("This neutral news suggests limited immediate market impact.")
            
            # Add volatility analysis
            if sentiment_indicators["volatility"] > 0:
                explanation.append("The market may experience increased volatility in response to this news.")
            
            # Add trend analysis
            if sentiment_indicators["trend"] > 0:
                explanation.append("This news may influence the overall market trend.")
            
            # Add trading signals
            explanation.append("\nTrading Signals:")
            if sentiment == "Positive":
                explanation.append("- Consider entering long positions or holding existing ones")
                explanation.append("- Look for buying opportunities on pullbacks")
            elif sentiment == "Negative":
                explanation.append("- Consider taking profits on long positions")
                explanation.append("- Look for short-selling opportunities")
            else:
                explanation.append("- Maintain current positions")
                explanation.append("- Wait for clearer market signals")
            
            # Add risk management advice
            explanation.append("\nRisk Management:")
            explanation.append("- Always use stop-loss orders")
            explanation.append("- Consider position sizing based on risk tolerance")
            explanation.append("- Monitor market conditions closely")
            
            return "\n".join(explanation)
            
    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
        # Ultimate fallback
        if sentiment == "Positive":
            return "This positive news might have a favorable impact on the asset's price. Consider this as a potential bullish signal, but always do your own research before making investment decisions."
        elif sentiment == "Negative":
            return "This negative news might have an unfavorable impact on the asset's price. Consider this as a potential bearish signal, but always do your own research before making investment decisions."
        else:
            return "This neutral news might not significantly impact the asset's price. The market could move either way, so always do your own research before making investment decisions."


@st.cache_data(ttl=3600)  # Cache for 1 hour
def scrape_news():
    """
    Scrape business/financial headlines from multiple news websites.
    Returns a list of dictionaries containing the source and headline.
    """
    news_data = []

    # Scrape Yahoo Finance (Stock Market News)
    try:
        # URL of the CNBC World News page
        yahoo_url = "https://finance.yahoo.com"

        # Add a User-Agent header to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        # Send a GET request to the URL
        response = requests.get(yahoo_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all headline elements using the correct tag and class
            headlines = soup.find_all('a', class_='subtle-link')
            
            # Loop through the first 10 headlines and extract the text
            for headline in headlines[:10]:  # Limit to 10 headlines
                text = headline.text.strip()  # Extract and clean the headline text
                if any(keyword in text.lower() for keyword in ["stock", "market", "business", "trading", "crypto"]):
                    news_data.append({
                        'source': 'YAHOO',
                        'headline': text
                    })
        else:
            st.write(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error scraping Yahoo Finance: {e}")

    # Scrape Reuters (Business Section)
    try:
        reuters_url = "https://www.reuters.com/business/"
        response = requests.get(reuters_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = soup.find_all('h2')  # Adjust based on the website's structure
        for headline in headlines[:10]:  # Limit to 10 headlines
            text = headline.text.strip()
            if any(keyword in text.lower() for keyword in ["stock", "market", "business", "trading", "crypto"]):
                news_data.append({
                    'source': 'Reuters',
                    'headline': text
                })
    except Exception as e:
        print(f"Error scraping Reuters: {e}")

    # Scrape CNBC (Markets Section)
    try:
        # URL of the CNBC World News page
        cnbc_url = "https://www.cnbc.com/world/"
        
        # Add a User-Agent header to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        # Send a GET request to the URL
        response = requests.get(cnbc_url, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all headline elements using the correct tag and class
            headlines = soup.find_all('a', class_='LatestNews-headline')
            
            # Loop through the first 10 headlines and extract the text
            for headline in headlines[:10]:  # Limit to 10 headlines
                text = headline.text.strip()  # Extract and clean the headline text
                if any(keyword in text.lower() for keyword in ["stock", "market", "business", "trading", "crypto"]):
                    news_data.append({
                        'source': 'CNBC',
                        'headline': text
                    })
        else:
            st.write(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error scraping CNBC: {e}")

    return news_data


def sentimentTT():
    st.header("Market Sentiment Analysis")
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3 = st.tabs(["Latest News", "Detailed Analysis", "Overall Market Sentiment"])
    
    with tab1:
        st.subheader("Latest Financial News")
        with st.spinner("Fetching latest business news..."):
            try:
                news_data = scrape_news()
                if news_data:
                    # Display headlines in a grid layout
                    cols_per_row = 3
                    rows = [news_data[i:i + cols_per_row] for i in range(0, len(news_data), cols_per_row)]
                    for row in rows:
                        cols = st.columns(cols_per_row)
                        for i, news in enumerate(row):
                            if i < len(cols):
                                with cols[i]:
                                    st.markdown(f"""
                                    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
                                        <p><strong>{news['source']}</strong></p>
                                        <p>{news['headline']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.warning("No business news headlines found.")
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
                st.warning("Could not fetch news headlines.")
        
        # Add custom news analysis section in the Latest News tab
        st.subheader("Analyze Custom News")
        headline = st.text_area("Enter a financial news headline:")
        if st.button("Analyze Custom Headline"):
            if headline:
                try:
                    with st.spinner("Analyzing sentiment..."):
                        sentiment, scores = predict_sentiment(headline)
                        
                        # Display results in a visually appealing way
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"Predicted Sentiment: **{sentiment}**")
                            
                            # Create a bar chart for sentiment scores
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(scores.keys()),
                                    y=list(scores.values()),
                                    marker_color=['red', 'gray', 'green']
                                )
                            ])
                            fig.update_layout(
                                title="Sentiment Distribution",
                                xaxis_title="Sentiment",
                                yaxis_title="Score",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col2:
                            st.write("### Sentiment Scores")
                            for sentiment_type, score in scores.items():
                                st.write(f"{sentiment_type}: {score:.2%}")
                            
                        # Generate and display market impact analysis
                        with st.spinner("Generating market impact analysis..."):
                            explanation = generate_explanation(headline, sentiment)
                            st.write("### Market Impact Analysis")
                            st.markdown(explanation)
                            
                            # Add trading signals based on sentiment
                            st.write("### Trading Signals")
                            if sentiment == "Positive":
                                st.success("**Trading Signal**: Consider a bullish position or hold existing long positions")
                            elif sentiment == "Negative":
                                st.error("**Trading Signal**: Consider a bearish position or take profits on long positions")
                            else:
                                st.info("**Trading Signal**: Maintain current positions or wait for clearer signals")
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {str(e)}")
                    st.warning("There was an issue with the sentiment analysis. Please try again later.")
            else:
                st.warning("Please enter some news content")

    with tab2:
        st.subheader("Detailed News Analysis")
        try:
            if news_data:
                # Create a selectbox for news selection
                headline_options = [f"{news['source']}: {news['headline']}" for news in news_data]
                selected_headline_idx = st.selectbox("Choose a headline to analyze:", range(len(headline_options)), format_func=lambda x: headline_options[x])
                
                if st.button("Analyze Selected Headline"):
                    try:
                        article_headline = news_data[selected_headline_idx]["headline"]
                        with st.spinner("Analyzing sentiment..."):
                            # Predict sentiment and get scores
                            predicted_sentiment, sentiment_scores = predict_sentiment(article_headline)  
                            
                            # Display sentiment results in a more visually appealing way
                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(f"Predicted Sentiment: **{predicted_sentiment}**")
                                
                                # Create a bar chart for sentiment scores
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=list(sentiment_scores.keys()),
                                        y=list(sentiment_scores.values()),
                                        marker_color=['red', 'gray', 'green']
                                    )
                                ])
                                fig.update_layout(
                                    title="Sentiment Distribution",
                                    xaxis_title="Sentiment",
                                    yaxis_title="Score",
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.write("### Sentiment Scores")
                                for sentiment, score in sentiment_scores.items():
                                    st.write(f"{sentiment}: {score:.2%}")
                            
                            # Generate and display market impact analysis
                            with st.spinner("Generating market impact analysis..."):
                                market_impact_analysis = generate_explanation(article_headline, predicted_sentiment)
                                st.write("### Market Impact Analysis")
                                st.markdown(market_impact_analysis)
                                
                                # Add trading signals based on sentiment
                                st.write("### Trading Signals")
                                if predicted_sentiment == "Positive":
                                    st.success("**Trading Signal**: Consider a bullish position or hold existing long positions")
                                elif predicted_sentiment == "Negative":
                                    st.error("**Trading Signal**: Consider a bearish position or take profits on long positions")
                                else:
                                    st.info("**Trading Signal**: Maintain current positions or wait for clearer signals")
                    except Exception as e:
                        st.error(f"Error analyzing sentiment: {str(e)}")
                        st.warning("There was an issue with the sentiment analysis. Please try again later.")
            else:
                st.warning("No news data available for analysis.")
        except Exception as e:
            st.error(f"Error in detailed analysis: {str(e)}")
    
    with tab3:
        st.subheader("Overall Market Sentiment")
        try:
            if news_data:
                # Analyze sentiment for each news item
                sentiments = []
                for news in news_data:
                    sentiment, scores = predict_sentiment(news['headline'])
                    sentiments.append({
                        'source': news['source'],
                        'headline': news['headline'],
                        'sentiment': sentiment,
                        'scores': scores
                    })
                
                # Calculate overall news sentiment
                positive_count = sum(1 for s in sentiments if s['sentiment'] == 'Positive')
                negative_count = sum(1 for s in sentiments if s['sentiment'] == 'Negative')
                neutral_count = sum(1 for s in sentiments if s['sentiment'] == 'Neutral')
                
                # Display sentiment distribution
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive News", positive_count)
                with col2:
                    st.metric("Negative News", negative_count)
                with col3:
                    st.metric("Neutral News", neutral_count)
                
                # Calculate weighted sentiment scores
                total_sentiment_score = 0
                total_weight = 0
                
                for sentiment_data in sentiments:
                    # Weight based on source reliability
                    weight = 1.0
                    if sentiment_data['source'] == 'Reuters':
                        weight = 1.2
                    elif sentiment_data['source'] == 'CNBC':
                        weight = 1.1
                    
                    # Convert sentiment to numerical score
                    sentiment_score = (
                        sentiment_data['scores']['Positive'] * 1 +
                        sentiment_data['scores']['Negative'] * -1 +
                        sentiment_data['scores']['Neutral'] * 0
                    )
                    
                    total_sentiment_score += sentiment_score * weight
                    total_weight += weight
                
                # Calculate final sentiment score
                final_sentiment_score = total_sentiment_score / total_weight if total_weight > 0 else 0
                
                # Determine market mood
                if final_sentiment_score > 0.2:
                    market_mood = "Bullish"
                    mood_color = "green"
                elif final_sentiment_score < -0.2:
                    market_mood = "Bearish"
                    mood_color = "red"
                else:
                    market_mood = "Neutral"
                    mood_color = "gray"
                
                # Display market mood
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid {mood_color}; border-radius: 10px;">
                    <h2>Current Market Mood: <span style="color: {mood_color};">{market_mood}</span></h2>
                    <p>Sentiment Score: {final_sentiment_score:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display sentiment distribution chart
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Positive', 'Negative', 'Neutral'],
                        values=[positive_count, negative_count, neutral_count],
                        hole=0.4,
                        marker_colors=['green', 'red', 'gray']
                    )
                ])
                fig.update_layout(title="Overall Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Market sentiment insights
                st.subheader("Market Sentiment Insights")
                if market_mood == "Bullish":
                    st.success("""
                    **Bullish Market Indicators:**
                    - Positive news sentiment outweighs negative sentiment
                    - Market confidence appears strong
                    - Consider looking for buying opportunities
                    """)
                elif market_mood == "Bearish":
                    st.error("""
                    **Bearish Market Indicators:**
                    - Negative news sentiment outweighs positive sentiment
                    - Market confidence appears weak
                    - Consider defensive positions or short opportunities
                    """)
                else:
                    st.info("""
                    **Neutral Market Indicators:**
                    - Mixed sentiment signals
                    - Market direction unclear
                    - Consider maintaining current positions
                    """)
                
                # Trading recommendations
                st.subheader("Trading Recommendations")
                if market_mood == "Bullish":
                    st.success("""
                    **Recommended Actions:**
                    - Look for buying opportunities
                    - Consider increasing long positions
                    - Set appropriate stop-loss levels
                    """)
                elif market_mood == "Bearish":
                    st.error("""
                    **Recommended Actions:**
                    - Consider taking profits on long positions
                    - Look for short-selling opportunities
                    - Implement strict risk management
                    """)
                else:
                    st.info("""
                    **Recommended Actions:**
                    - Maintain current positions
                    - Wait for clearer market signals
                    - Focus on risk management
                    """)
            else:
                st.warning("Insufficient data for market sentiment analysis")
        except Exception as e:
            st.error(f"Error calculating overall market sentiment: {str(e)}")

def compare_models():
    st.header("Compare Models")
    
    # Add date range selection
    st.write("Select a date range:")
    start_dat = st.date_input("Start Date", st.session_state.data.index.min().date())
    end_dat = st.date_input("End Date", st.session_state.data.index.max().date())
    
    if start_dat > end_dat:
        st.error("Start Date cannot be after End Date.")
        return
    
    # Filter data based on selected date range
    filtered_data = st.session_state.data[(st.session_state.data.index >= pd.Timestamp(start_dat)) & 
                                        (st.session_state.data.index <= pd.Timestamp(end_dat))]
    
    num_dayZ = st.number_input('How many days forecast for comparison?', value=5, min_value=1)
    num_dayZ = int(num_dayZ)

    if st.button('Run Comparison'):
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'ExtraTreesRegressor': ExtraTreesRegressor(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'XGBoostRegressor': XGBRegressor()
        }
        
        # Data preparation
        if st.session_state.data is None:
            st.error("No data available for comparison.")
            return
        
        df = filtered_data[['Close']].copy()
        df['Future'] = df['Close'].shift(-num_dayZ)
        x_data = df[['Close']].values
        y_data = df['Future'].values
        
        x_scaled = st.session_state.scaler.fit_transform(x_data)
        x_forecast = x_scaled[-num_dayZ:]
        x_train = x_scaled[:-num_dayZ]
        y_train = y_data[:-num_dayZ]

        x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
            x_train, y_train, test_size=0.2, random_state=7)

        results = {'Model': [], 'R² Score': [], 'MAE': [], 'MAPE (%)': []}

        for name, model in models.items():
            model.fit(x_train_split, y_train_split)
            predictions = model.predict(x_test_split)
            r2 = r2_score(y_test_split, predictions)
            mae = mean_absolute_error(y_test_split, predictions)
            
            # MAPE calculation with handling for zero values
            non_zero_mask = y_test_split != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_test_split[non_zero_mask] - predictions[non_zero_mask]) / y_test_split[non_zero_mask])) * 100
            else:
                mape = np.nan  # If all values are zero, MAPE is undefined

            results['Model'].append(name)
            results['R² Score'].append(r2)
            results['MAE'].append(mae)
            results['MAPE (%)'].append(mape)

        # Display results
        st.subheader("Model Performance Metrics")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Visualize the results
        st.subheader("Model Comparison Chart")
        fig = px.bar(
            results_df,
            x='Model',
            y=['R² Score', 'MAE', 'MAPE (%)'],
            barmode='group',
            title='Comparison of Model Performance',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig)

def advanced_model_visualization():
    st.header("Advanced Model Visualization")
    
    if st.session_state.data is None:
        st.error("No data available for visualization. Please upload data or select a stock first.")
        return
    
    # Add date range selection
    st.write("Select a date range:")
    start_dat = st.date_input("Start Date", st.session_state.data.index.min().date())
    end_dat = st.date_input("End Date", st.session_state.data.index.max().date())

    if start_dat > end_dat:
        st.error("Start Date cannot be after End Date.")
        return
    
    # Filter data based on selected date range
    filtered_data = st.session_state.data[(st.session_state.data.index >= pd.Timestamp(start_dat)) & 
                                        (st.session_state.data.index <= pd.Timestamp(end_dat))]
    
    # Check if the data has the required columns
    if 'Close' not in filtered_data.columns:
        st.error("The data must contain a 'Close' column for visualization.")
        return
    
    # User inputs
    num_days = st.slider("Number of days to forecast", min_value=1, max_value=30, value=7)
    train_size = st.slider("Training data size (%)", min_value=50, max_value=95, value=80)
    
    # Model selection with multiselect
    available_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'XGBoost': XGBRegressor(n_estimators=100)
    }
    
    selected_models = st.multiselect(
        "Select models to compare",
        list(available_models.keys()),
        default=["Linear Regression", "Random Forest"]
    )
    
    if not selected_models:
        st.warning("Please select at least one model to visualize.")
        return
    
    if st.button("Generate Visualization"):
        with st.spinner("Training models and generating visualizations..."):
            try:
                # Data preparation
                df = filtered_data[['Close']].copy()
                
                # Feature engineering
                df['MA7'] = df['Close'].rolling(window=7).mean()
                df['MA14'] = df['Close'].rolling(window=14).mean()
                df['MA30'] = df['Close'].rolling(window=30).mean()
                
                # Calculate daily returns
                df['Returns'] = df['Close'].pct_change()
                
                # Calculate volatility (rolling standard deviation)
                df['Volatility'] = df['Returns'].rolling(window=14).std()
                
                # Target variable - future price
                df['Target'] = df['Close'].shift(-num_days)
                
                # Drop NaN values
                df = df.dropna()
                
                if len(df) < 30:
                    st.error("Not enough data for visualization after feature engineering. Need at least 30 data points.")
                    return
                
                # Features and target
                X = df[['Close', 'MA7', 'MA14', 'MA30', 'Returns', 'Volatility']]
                y = df['Target']
                
                # Train-test split
                split_idx = int(len(df) * (train_size/100))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Scale the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models and make predictions
                results = {}
                metrics = {'Model': [], 'R² Score': [], 'MAE': [], 'MAPE (%)': []}
                
                for model_name in selected_models:
                    model = available_models[model_name]
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Store predictions
                    results[model_name] = y_pred
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # MAPE calculation with handling for zero values
                    non_zero_mask = y_test != 0
                    if np.any(non_zero_mask):
                        mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
                    else:
                        mape = np.nan
                    
                    # Store metrics
                    metrics['Model'].append(model_name)
                    metrics['R² Score'].append(r2)
                    metrics['MAE'].append(mae)
                    metrics['MAPE (%)'].append(mape)
                
                # Create metrics dataframe
                metrics_df = pd.DataFrame(metrics)
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                st.dataframe(metrics_df)
                
                # Create visualization dataframe
                viz_df = pd.DataFrame({'Actual': y_test})
                for model_name, preds in results.items():
                    viz_df[model_name] = preds
                
                # Plot actual vs predicted
                st.subheader("Actual vs Predicted Prices")
                
                # Line chart
                fig1 = px.line(
                    viz_df, 
                    title="Model Predictions Comparison",
                    labels={'value': 'Price', 'index': 'Time'}
                )
                st.plotly_chart(fig1)
                
                # Calculate prediction errors
                error_df = pd.DataFrame()
                for model_name in selected_models:
                    error_df[model_name] = viz_df['Actual'] - viz_df[model_name]
                
                # Plot prediction errors
                st.subheader("Prediction Errors")
                fig2 = px.line(
                    error_df,
                    title="Prediction Errors Over Time",
                    labels={'value': 'Error', 'index': 'Time'}
                )
                # Add a horizontal line at y=0
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig2)
                
                # Plot error distribution
                st.subheader("Error Distribution")
                fig3 = go.Figure()
                for model_name in selected_models:
                    fig3.add_trace(go.Histogram(
                        x=error_df[model_name],
                        name=model_name,
                        opacity=0.7,
                        nbinsx=30
                    ))
                fig3.update_layout(
                    title="Error Distribution by Model",
                    xaxis_title="Prediction Error",
                    yaxis_title="Frequency",
                    barmode='overlay'
                )
                st.plotly_chart(fig3)
                
                # Feature importance for tree-based models
                tree_models = ['Random Forest', 'Extra Trees', 'XGBoost']
                tree_models_selected = [m for m in selected_models if m in tree_models]
                
                if tree_models_selected:
                    st.subheader("Feature Importance")
                    
                    # Create feature importance DataFrame with proper index
                    feature_importance_df = pd.DataFrame({
                        'Feature': X.columns
                    })
                    
                    for model_name in tree_models_selected:
                        model = available_models[model_name]
                        if hasattr(model, 'feature_importances_'):
                            feature_importance_df[model_name] = model.feature_importances_
                    
                    # Plot feature importance
                    if not feature_importance_df.empty and len(feature_importance_df.columns) > 1:
                        fig4 = px.bar(
                            feature_importance_df.melt(id_vars=['Feature'], var_name='Model', value_name='Importance'),
                            x='Feature',
                            y='Importance',
                            color='Model',
                            title="Feature Importance by Model",
                            barmode='group'
                        )
                        st.plotly_chart(fig4)
                
                # Future forecast
                st.subheader(f"Future {num_days}-Day Forecast")
                
                # Get the most recent data for prediction
                latest_data = X.iloc[-1:].values
                latest_data_scaled = scaler.transform(latest_data)
                
                # Create forecast results dictionary
                forecast_results = {}
                for model_name in selected_models:
                    model = available_models[model_name]
                    forecast = model.predict(latest_data_scaled)[0]
                    forecast_results[model_name] = forecast
                
                # Create forecast DataFrame with proper structure
                forecast_df = pd.DataFrame({
                    'Model': list(forecast_results.keys()),
                    'Forecast': list(forecast_results.values())
                })
                
                # Add current price and calculate changes
                current_price = df['Close'].iloc[-1]
                forecast_df['Current Price'] = current_price
                forecast_df['Change'] = forecast_df['Forecast'] - forecast_df['Current Price']
                forecast_df['Change (%)'] = (forecast_df['Change'] / forecast_df['Current Price']) * 100
                
                # Display forecast
                st.dataframe(forecast_df)
                
                # Plot forecast comparison
                fig5 = px.bar(
                    forecast_df,
                    x='Model',
                    y='Change (%)',
                    title=f"{num_days}-Day Forecast: Expected Price Change (%)",
                    color='Change (%)',
                    color_continuous_scale=['red', 'gray', 'green'],
                    range_color=[-10, 10]
                )
                st.plotly_chart(fig5)
            except Exception as e:
                st.error(f"An error occurred during visualization: {str(e)}")
                st.error("Please check your data format and try again.")

def main():
    st.title('Trading Price Predictor')
    
    option = st.sidebar.radio(
        "Choose how to proceed:",
        ("Continue without uploading", "Upload a CSV file")
    )
    
    data = None
    start_date = None
    end_date = None
    ticker_symbol = None
    uploaded_file = None
    
    if option == "Upload a CSV file":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            try:
                fwd = pd.read_csv(uploaded_file)
                if 'Date' not in fwd.columns:
                    st.error("The CSV file must contain a 'Date' column.")
                    return
                fwd['Date'] = pd.to_datetime(fwd['Date'])
                fwd['Date'] = fwd['Date'].dt.tz_localize(None)
                filtered_data = fwd.set_index('Date')
                data = filtered_data
                st.session_state.data = data
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
        else:
            st.warning("Please upload a CSV file to proceed.")
    else:
        input_method = st.sidebar.radio(
            "Choose how to enter stock symbol:",
            ("Select from list", "Type manually")
        )
        
        if input_method == "Select from list":
            ticker_symbol = st.sidebar.selectbox('Select Stock Symbol', stocks)
        else:
            ticker_symbol = st.sidebar.text_input('Enter Stock Symbol', value='AAPL')
            if ticker_symbol not in stocks:
                st.sidebar.warning(f"Note: {ticker_symbol} is not in our predefined list. Make sure it's a valid symbol.")
        
        start_date = st.sidebar.date_input('Start Date', date(2020, 1, 1))
        end_date = st.sidebar.date_input('End Date', date.today())
        data = download_data(ticker_symbol, start_date, end_date)
        st.session_state.data = data

    # Main menu
    menu_option = st.sidebar.selectbox(
        'Select Feature',
        [
            'Visualize Technical Indicators',
            'Recent Data',
            'Predict',
            'Compare Models',
            'Live Chart',
            'Sentiment Analysis',
            'Advanced Model Visualization'
        ]
    )

    # Display selected feature
    if menu_option == 'Visualize Technical Indicators':
        tech_indicators(data)
    elif menu_option == 'Recent Data':
        if data is not None:
            st.header('Recent Data')
            st.dataframe(data.tail(100))
        else:
            st.error("No data available to display")
    elif menu_option == 'Predict':
        if data is not None:
            predict()
        else:
            st.error("No data available for prediction")
    elif menu_option == 'Compare Models':
        if data is not None:
            compare_models()
        else:
            st.error("No data available for model comparison")
    elif menu_option == 'Live Chart':
        tradingview()
    elif menu_option == 'Sentiment Analysis':
        sentimentTT()
    elif menu_option == 'Advanced Model Visualization':
        advanced_model_visualization()

if __name__ == '__main__':
    main()
