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

st.set_page_config(page_title='Stock Price Predictor', layout='wide')
# st.title('Stock Price Predictor')

# Define the base URLs for accessing the API
BASE_URLS = [
    # "https://api-handler-ddc-free-api.hf.space/v2"
    "https://api.sree.shop/v1"
]

# Initialize the OpenAI client with a specific base URL and API key
client = OpenAI(
    base_url=BASE_URLS[0],  # Using the first URL in the list
    # api_key="DDC-Free-For-Subscribers-YT-@DevsDoCode"
    api_key="ddc-dJ72dJ85WuKMrINtNG6Rd9E4lHhxXmWMwJeYk4WlxSYhSPPFV3"
)
# Load FinBERT Model and Tokenizer
@st.cache_resource
def load_finbert():
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    modeL = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, modeL

tokenizer, modeL = load_finbert()

# Page configuration
# st.set_page_config(page_title='Stock Price Predictor', layout='wide')
# st.title('Stock Price Predictor')

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

# Sidebar inputs
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'BTC-USD',
 'ETH-USD',
 'USDT-USD',
 'BNB-USD',
 'SOL-USD',
 'USDC-USD',
 'STETH-USD',
 'XRP-USD',
 'DOGE-USD',
 'TON11419-USD',
 'ADA-USD',
 'SHIB-USD',
 'AVAX-USD',
 'TRX-USD',
 'WTRX-USD',
 'WBTC-USD',
 'DOT-USD',
 'BCH-USD',
 'LINK-USD',
 'NEAR-USD',
 'MATIC-USD',
 'HDFCBANK.NS')
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
    if data is None:
        st.error("No data available for technical analysis")
        return

    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', 
                      ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA', 'Ichimoku'])

    try:
        # Ensure 'Close', 'High', 'Low' columns are squeezed into 1D
        close_series = data['Close'].squeeze() if data['Close'].ndim > 1 else data['Close']
        high_series = data['High'].squeeze() if data['High'].ndim > 1 else data['High']
        low_series = data['Low'].squeeze() if data['Low'].ndim > 1 else data['Low']

        def plot_close_price():
            fig = px.line(x=data.index, y=close_series, title='Closing Price')
            st.plotly_chart(fig)

        def plot_bollinger_bands():
            bb_indicator = BollingerBands(close_series, window=20, window_dev=2)
            bb = pd.DataFrame({
                'Close': close_series,
                'bb_h': bb_indicator.bollinger_hband(),
                'bb_l': bb_indicator.bollinger_lband()
            }, index=data.index)
            fig = px.line(bb, x=bb.index, y=['Close', 'bb_h', 'bb_l'], title='Bollinger Bands')
            st.plotly_chart(fig)

        def plot_ichimoku():
            ichimoku = IchimokuIndicator(high=high_series, low=low_series, window1=9, window2=26, window3=52)
            ichimoku_data = pd.DataFrame({
                'Close': close_series,
                'ichimoku_a': ichimoku.ichimoku_a(),
                'ichimoku_b': ichimoku.ichimoku_b()
            }, index=data.index)
            fig = px.line(ichimoku_data, x=ichimoku_data.index, y=['Close', 'ichimoku_a', 'ichimoku_b'], title='Ichimoku Cloud')
            st.plotly_chart(fig)

        # Plot the selected indicator
        if option == 'Close':
            plot_close_price()
        elif option == 'BB':
            plot_bollinger_bands()
        elif option == 'MACD':
            macd = MACD(close_series).macd()
            plot_chart(pd.DataFrame({'MACD': macd}, index=data.index).squeeze(), 'MACD', 'Moving Average Convergence Divergence')
        elif option == 'RSI':
            rsi = RSIIndicator(close_series).rsi()
            plot_chart(pd.DataFrame({'RSI': rsi}, index=data.index).squeeze(), 'RSI', 'Relative Strength Index')
        elif option == 'SMA':
            sma = SMAIndicator(close_series, window=14).sma_indicator()
            plot_chart(pd.DataFrame({'SMA': sma}, index=data.index).squeeze(), 'SMA', 'Simple Moving Average')
        elif option == 'EMA':
            ema = EMAIndicator(close_series, window=14).ema_indicator()
            plot_chart(pd.DataFrame({'EMA': ema}, index=data.index).squeeze(), 'EMA', 'Exponential Moving Average')
        elif option == 'Ichimoku':
            plot_ichimoku()

    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        st.error("Please make sure your data contains the required columns (Close, High, Low)")

def predict():
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
        elif model_choice == 'RandomForestRegressor':
            engine = RandomForestRegressor()
        elif model_choice == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
        elif model_choice == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
        else:
            engine = XGBRegressor()
        
        # Run the model engine with the selected model and forecast days
        model_engine(engine, num_days)


def model_engine(model, forecast_days):
    # Copy only the 'Close' column for modeling to avoid the chained assignment warning
    df = st.session_state.data[['Close']].copy()
    
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
    mape = np.mean(np.abs((y_test_split - predictions) / y_test_split)) * 100  # MAPE calculation
    
    # Display evaluation metrics
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Forecasting future prices
    future_predictions = model.predict(x_forecast)
    
    st.subheader(f"{forecast_days}-Day Price Forecast")
    for day, price in enumerate(future_predictions, start=1):
        st.write(f"Day {day}: ${price:.2f}")


def stock_info(symbol):
    st.header('Stock Information')
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"*Company Name:* {info.get('longName', 'N/A')}")
            st.write(f"*Sector:* {info.get('sector', 'N/A')}")
            st.write(f"*Industry:* {info.get('industry', 'N/A')}")
        with col2:
            st.write(f"*Market Cap:* {info.get('marketCap', 'N/A')}")
            st.write(f"*Dividend Yield:* {info.get('dividendYield', 'N/A')}")
            st.write(f"*P/E Ratio:* {info.get('trailingPE', 'N/A')}")
    except Exception as e:
        st.error(f"Error fetching stock info: {str(e)}")

def tradingview():
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

def chat():
    st.title("OpenAI Chatbot")
    # Chatbot feature
    user_input = st.text_input("Ask something:")
    if st.button("Get Response"):
        if user_input.strip():
            try:
                completion = client.chat.completions.create(
                    model="provider-3/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )

                # Access the content from the ChatCompletionMessage object
                bot_reply = completion.choices[0].message.content
                st.write(f"**Chatbot**: {bot_reply}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

# Function to Predict Sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = modeL(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    sentiments = ["Negative", "Neutral", "Positive"]
    sentiment_scores = {sentiments[i]: probs[0][i].item() for i in range(len(sentiments))}
    predicted_sentiment = sentiments[torch.argmax(probs)]
    return predicted_sentiment, sentiment_scores

# Function to Generate Context-Aware Explanation
def generate_explanation(headline, sentiment):
    prompt = f"""
    The following is a financial news headline: "{headline}"
    Sentiment analysis indicates that the sentiment is {sentiment}.
    
    Based on this information, provide an explanation of how this news might impact the stock or forex or crypto or futures&options market or the specific asset mentioned, provide a theoretical signal whether user should buy/sell or long/short the given asset. Consider historical market trends, potential investor reactions, and the language in the news.
    """
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {"role": "system", "content": "You are a financial expert providing context-aware market analysis."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1000,
    )
    return response.choices[0].message.content


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
    st.title("Business News Headlines")
    
    # Automatically fetch and display business news headlines
    with st.spinner("Fetching latest business news..."):
        news_data = scrape_news()
        if news_data:
            st.subheader("Latest Business News Headlines")
            
            # Display headlines in a grid layout
            cols_per_row = 3  # Number of columns per row
            rows = [news_data[i:i + cols_per_row] for i in range(0, len(news_data), cols_per_row)]
            for row in rows:
                cols = st.columns(cols_per_row)
                for col, news in zip(cols, row):
                    with col:
                        st.markdown(f"""
                        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
                            <p><strong>{news['source']}</strong></p>
                            <p>{news['headline']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            st.subheader("Sentiment & Signal")

            for i in range(len(news_data)):
                article_headline = news_data[i]["headline"]  
                
                if article_headline:  
                    with st.spinner("Analyzing sentiment..."):
                        # Predict sentiment and get scores
                        predicted_sentiment, sentiment_scores = predict_sentiment(article_headline)  
                        st.success(f"Predicted Sentiment: **{predicted_sentiment}**")
                        
                        # Display sentiment scores
                        st.write("### Sentiment Scores")
                        st.json(sentiment_scores)  
                        
                        # Generate explanation for market impact
                        with st.spinner("Generating explanation..."):
                            market_impact_analysis = generate_explanation(article_headline, predicted_sentiment)  
                            st.write("### Market Impact Analysis")
                            st.write(market_impact_analysis)
                else:
                    st.warning("Please enter some news content")
        else:
            st.warning("No business news headlines found.")
        
    # Text input for user to enter news
    headline = st.text_area("Enter a financial news headline:")
    if st.button("Analyze Sentiment"):
        if headline:
            with st.spinner("Analyzing sentiment..."):
                sentiment, scores = predict_sentiment(headline)
                st.success(f"Predicted Sentiment: **{sentiment}**")
                # Display sentiment scores
                st.write("### Sentiment Scores")
                st.json(scores)
                # Generate Explanation
                with st.spinner("Generating explanation..."):
                    explanation = generate_explanation(headline, sentiment)
                    st.write("### Market Impact Analysis")
                    st.write(explanation)
        else:
            st.warning("Please enter some news content")

def plot_model_predictions(end_date, option, ticker_symbo):
    # st.write(option)
    if option == "Upload a CSV file":
        ticker_symbo = st.text_input("Enter symbol name")
    
    st.header("Model Predictions vs Actual Values")
    num_dayss = st.number_input('How many days forecast for comparison?', value=5, min_value=1)
    num_dayss = int(num_dayss)
    # days = min(num_dayss, flag)
    if st.button('Run Comparison'):
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'ExtraTreesRegressor': ExtraTreesRegressor(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'XGBoostRegressor': XGBRegressor()
        }

        # Check if data exists
        if st.session_state.data is None:
            st.error("No data available for comparison.")
            return

        # Data preparation
        df = st.session_state.data[['Close']].copy()
        df['Future'] = df['Close'].shift(-num_dayss)

        x_data = df[['Close']].values
        y_data = df['Future'].values
        x_scaled = st.session_state.scaler.fit_transform(x_data)

        # Prepare data for forecasting and training
        x_forecast = x_scaled[-num_dayss:]
        x_train = x_scaled[:-num_dayss]
        y_train = y_data[:-num_dayss]

        x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
            x_train, y_train, test_size=0.2, random_state=7)

        results = {'Model': [], 'R² Score': [], 'MAE': [], 'MAPE (%)': []}
        predictions_dict = {}

        # Prepare data for visualization
        today = datetime.now()
        future_dates = [end_date + timedelta(days=i) for i in range(1, num_dayss + 1)]
        actual_values = df['Close'][-num_dayss:].values
        flag = future_dates[len(future_dates)-1]
        flag += timedelta(days=1)

        if flag < today.date():
            arr = download_data(ticker_symbo, future_dates[0], flag)
            full_date_range = pd.date_range(start=arr.index.min(), end=arr.index.max(), freq='D')
            # st.write(actual_values)
            # Reindex the DataFrame to include all days
            arr = arr.reindex(full_date_range)
            st.dataframe(arr.tail(100))
            actual_values = arr['Close'].values
            # st.dataframe(actual_values)
            # st.write(arr)
        else: 
            actual_values = None

        # Model training and evaluation
        for name, model in models.items():
            model.fit(x_train_split, y_train_split)
            predictions = model.predict(x_test_split)

            r2 = r2_score(y_test_split, predictions)
            mae = mean_absolute_error(y_test_split, predictions)
            mape = np.mean(np.abs((y_test_split - predictions) / y_test_split)) * 100  # Avoid zero division errors

            results['Model'].append(name)
            results['R² Score'].append(r2)
            results['MAE'].append(mae)
            results['MAPE (%)'].append(mape)

            # Generate future predictions
            future_predictions = model.predict(x_forecast)
            # Ensure future_predictions matches actual_values length
            if actual_values is not None and len(future_predictions) > len(actual_values):
                future_predictions = future_predictions[:len(actual_values)]
            predictions_dict[name] = future_predictions

        # Display model metrics
        st.subheader("Model Performance Metrics")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        plot_data = pd.DataFrame({'Date': future_dates})
        
        if actual_values is not None and len(actual_values) == len(future_dates):
            plot_data['Actual'] = actual_values
        else:
            st.warning("Value Mismatch due to Market Holiday, Please Choose a different Date.")
            return

        for model_name, prediction in predictions_dict.items():
            plot_data[model_name] = prediction

        
        # Visualization of Actual vs Predicted
        st.subheader("Prediction vs Actual Values")
        fig = px.line(
            plot_data,
            x='Date',
            y=plot_data.columns[1:],  # Exclude Date from the y-axis selection
            markers=True,
            title='Actual vs Predicted Prices',
            labels={'value': 'Price', 'Date': 'Days'}
        )
        st.plotly_chart(fig)

def compare_models():
    st.header("Compare Models")
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
        
        df = st.session_state.data[['Close']].copy()
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
            mape = np.mean(np.abs((y_test_split - predictions) / y_test_split)) * 100  # MAPE calculation
            results['Model'].append(name)
            results['R² Score'].append(r2)
            results['MAE'].append(mae)
            results['MAPE (%)'].append(mape)
            # future_prediction = model.predict(x_forecast)
            # for day, price in enumerate(future_prediction, start=1):
            #     st.write(f"Day {day}: ${price:.2f}")

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

def main():
    option = st.sidebar.radio(
        "Choose how to proceed:",
        ("Continue without uploading", "Upload a CSV file")
    )
    if option == "Upload a CSV file":
            uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
            fwd = pd.read_csv(uploaded_file, parse_dates=["Date"])
            
            # Ensure the Date column is in datetime format and remove timezone if present
            fwd['Date'] = pd.to_datetime(fwd['Date']).dt.tz_localize(None)

            # Add a date range selector
            st.write("Select a date range:")
            start_dat = st.date_input("Start Date", fwd['Date'].min().date())
            end_dat = st.date_input("End Date", fwd['Date'].max().date())

            if start_dat > end_dat:
                st.error("Start Date cannot be after End Date.")
            else:
                # Filter the dataset based on the date range
                filtered_data = fwd[(fwd['Date'] >= pd.Timestamp(start_dat)) & (fwd['Date'] <= pd.Timestamp(end_dat))]
                data = pd.DataFrame(filtered_data)
                start_date = start_dat
                end_date = end_dat
                st.session_state.data = data

    else:
        ticker_symbol = st.sidebar.selectbox('Enter Stock Symbol', stocks)
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
            'Plot Models',
            'Stock Info',
            'Live Chart',
            'Sentiment Analysis',
            'AI Chatbot'
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
        predict()
    elif menu_option == 'Compare Models':
        compare_models()
    elif menu_option == 'Plot Models':
        if option == "Upload a CSV file":
            ticker_symb = "none"
            plot_model_predictions(end_date, option, ticker_symb)
        else:
            plot_model_predictions(end_date, option, ticker_symbol)
    elif menu_option == 'Stock Info':
        if option == "Upload a CSV file":
            st.warning("Local file uploaded")
        else:
            stock_info(ticker_symbol)
    elif menu_option == 'Live Chart':
        tradingview()
    elif menu_option == 'Sentiment Analysis':
        sentimentTT()
    elif menu_option == 'AI Chatbot':
        chat()

if __name__ == '__main__':
    main()
