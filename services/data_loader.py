# ./services/data_loader.py

import yfinance as yf
import pandas as pd
import asyncio
import time

# Function to load real-time stock data from Yahoo Finance
def load_real_time_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetches real-time stock data for the given symbol within a date range.
    Note: Yahoo Finance only allows up to 7 days worth of 1-minute data per request.
    :param symbol: Stock symbol to fetch data for.
    :param start: Start date for the historical data in 'YYYY-MM-DD' format.
    :param end: End date for the historical data in 'YYYY-MM-DD' format.
    :return: DataFrame containing historical stock data.

    :instantiation using function name
    :usage:
           from app/services/data_loader.py import load_real_time_data
           df = load_data("stock_name", "start_date", "end_date") # To store the dataframe object
    """
    stock = yf.Ticker(symbol)
    try:
        hist = stock.history(start=start, end=end, interval="1d")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    return hist

# Function to stream data asynchronously
async def stream_data(symbol: str, interval: int = 60, iterations: int = 5):
    """
    Implements real-time data streaming logic asynchronously.
    :param symbol: Stock symbol to stream data for.
    :param interval: Time interval in seconds between data fetches.
    :param iterations: Number of times to fetch data.

    :instantiation using await method
    :usage :
            import nest_asyncio
            nest_asyncio.apply()
            from app/services/data_loader.py import stream_data
            await stream_data("Stock_name","start_date","end_date")
    """
    for _ in range(iterations):
        stock = yf.Ticker(symbol)
        latest_data = stock.history(period="1d", interval="1m").tail(1)
        print(latest_data)
        await asyncio.sleep(interval)
    print("Streaming completed.")
    return latest_data