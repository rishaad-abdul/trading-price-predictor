import streamlit as st 
import pandas as pd
import plotly.express as px
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
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

# Define the base URLs for accessing the API
BASE_URLS = [
    "https://api-handler-ddc-free-api.hf.space/v2"
]

# Initialize the OpenAI client with a specific base URL and API key
client = OpenAI(
    base_url=BASE_URLS[0],  # Using the first URL in the list
    api_key="DDC-Free-For-Subscribers-YT-@DevsDoCode"
)
# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Page configuration
st.set_page_config(page_title='Stock Price Predictor', layout='wide')
st.title('Stock Price Predictor')

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
 'MATIC-USD')
st.sidebar.header('Input Parameters')
# st.selectbox('Select dataset for prediction', stocks)
ticker_symbol = st.selectbox('Enter Stock Symbol', stocks)
start_date = st.sidebar.date_input('Start Date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End Date', date.today())

@st.cache_data
def download_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
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
            plot_chart(pd.DataFrame({'MACD': macd}, index=data.index).squeeze(), 'MACD', 'MACD')
        elif option == 'RSI':
            rsi = RSIIndicator(close_series).rsi()
            plot_chart(pd.DataFrame({'RSI': rsi}, index=data.index).squeeze(), 'RSI', 'RSI')
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



# def sentiment_analysis(symbol):
#     st.header(f"News Sentiment for {symbol}")
#     try:
#         ticker = yf.Ticker(symbol)
#         news = ticker.news[:5] if ticker.news else []
        
#         if not news:
#             st.warning("No recent news found for this symbol")
#             return
            
#         analyzer = SentimentIntensityAnalyzer()
        
#         for article in news:
#             headline = article.get('title', '')
#             if headline:
#                 sentiment_score = analyzer.polarity_scores(headline)
#                 st.write(f"Headline: {headline}")
#                 st.write(f"Sentiment Score: {sentiment_score}")
#                 st.write("---")
                
#     except Exception as e:
#         st.error(f"Error analyzing sentiment: {str(e)}")

# def backtest_strategy(data):
#     if data is None:
#         st.error("No data available for backtesting")
#         return

#     st.header("Backtesting Strategy")
#     try:
#         # Ensure Close data is 1D
#         close_series = data['Close'].squeeze() if data['Close'].ndim > 1 else data['Close']
        
#         # Compute RSI and generate buy/sell signals
#         rsi = RSIIndicator(close_series).rsi()
#         buy_signals = rsi < 30
#         sell_signals = rsi > 70

#         # Extract buy and sell dates
#         buy_dates = data.index[buy_signals]
#         sell_dates = data.index[sell_signals]

#         # Display buy/sell signals
#         if len(buy_dates) > 0:
#             st.write("Buy Signals (RSI < 30):")
#             st.write(buy_dates.strftime('%Y-%m-%d').tolist())
#         else:
#             st.write("No buy signals found in this period")

#         if len(sell_dates) > 0:
#             st.write("Sell Signals (RSI > 70):")
#             st.write(sell_dates.strftime('%Y-%m-%d').tolist())
#         else:
#             st.write("No sell signals found in this period")

#     except Exception as e:
#         st.error(f"Error in backtesting: {str(e)}")


# def portfolio_performance(data):
#     if data is None:
#         st.error("No data available for portfolio analysis")
#         return

#     st.header("Portfolio Performance")
#     shares = st.number_input("Enter number of shares", min_value=1, value=100)
    
#     if st.button("Add to Portfolio"):
#         # Add stock to portfolio with current price
#         st.session_state.portfolio[ticker_symbol] = {
#             'shares': shares,
#             'price': data['Close'].iloc[-1] if not data['Close'].isnull().iloc[-1] else None
#         }

#     if st.session_state.portfolio:
#         for stock, info in st.session_state.portfolio.items():
#             current_price = data['Close'].iloc[-1] if not data['Close'].isnull().iloc[-1] else None
#             if info['price'] is not None and current_price is not None:
#                 roi = ((current_price - info['price']) / info['price'] * 100)
#                 st.write(f"{stock}:")
#                 st.write(f"Shares: {info['shares']}")
#                 st.write(f"Initial Price: ${info['price']:.2f}")
#                 st.write(f"Current Price: ${current_price:.2f}")
#                 st.write(f"ROI: {roi:.2f}%")
#             else:
#                 st.write(f"{stock}: Initial or current price data is unavailable.")
#             st.write("---")
#     else:
#         st.write("Portfolio is empty. Add some stocks to track performance.")

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
    
    # Display evaluation metrics
    # st.write(f"R^2 Score: {r2_score(y_test_split, predictions):.4f}")
    # st.write(f"Mean Absolute Error: {mean_absolute_error(y_test_split, predictions):.4f}")
    
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

def main():
    # Download data
    data = download_data(ticker_symbol, start_date, end_date)
    st.session_state.data = data
    
    # Main menu
    menu_option = st.sidebar.selectbox(
        'Select Feature',
        ['Visualize Technical Indicators', 'Recent Data', 'Predict', 'Stock Info', 'Live Chart', 'Sentiment Analysis', 'AI Chatbot']
    )
    
    # Display selected feature
    if menu_option == 'Visualize Technical Indicators':
        tech_indicators(data)
    elif menu_option == 'Recent Data':
        if data is not None:
            st.header('Recent Data')
            st.dataframe(data.tail(10))
        else:
            st.error("No data available to display")
    elif menu_option == 'Predict':
        predict()
    elif menu_option == 'Stock Info':
        stock_info(ticker_symbol)
    elif menu_option == 'Live Chart':
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
    elif menu_option == 'Sentiment Analysis':    
        st.title("News Sentiment Analysis")

        # Text input for user to enter news
        news = st.text_area("Enter news content:")
        if st.button("Analyze Sentiment"):
            if news.strip():
                # Perform sentiment analysis
                result = sentiment_pipeline(news)
                sentiment = result[0]['label']
                score = result[0]['score']

                # Display results
                st.write(f"**Sentiment**: {sentiment}")
                st.write(f"**Confidence Score**: {score:.2f}")
            else:
                st.warning("Please enter some news content.")
    elif menu_option == 'AI Chatbot':
        st.title("OpenAI Chatbot")

    # Chatbot feature
    user_input = st.text_input("Ask something:")
    if st.button("Get Response"):
        if user_input.strip():
            try:
                completion = client.chat.completions.create(
                    model="gpt-4",
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
    # elif menu_option == 'Backtest Strategy':
    #     backtest_strategy(data)
    # elif menu_option == 'Portfolio Performance':
    #     portfolio_performance(data)
    # elif menu_option == 'News Sentiment':
    #     sentiment_analysis(ticker_symbol)

if __name__ == '__main__':
    main()