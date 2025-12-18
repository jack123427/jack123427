
import time
from src.data_fetcher import get_historical_data, fetch_institutional_trading, fetch_margin_trading

def time_fetches():
    """
    Times each data fetching function to identify bottlenecks.
    """
    stock_code = '2330'
    days = 252 + 50 # Approximate number of days for a 1-year period

    print("--- Timing Data Fetching Functions ---")

    # 1. Time yfinance historical data fetching
    start_time = time.time()
    try:
        history_df = get_historical_data([stock_code], period="1y")
        print(f"✅ get_historical_data completed in: {time.time() - start_time:.2f} seconds")
        if history_df is None or history_df.empty:
            print("   -> Failed to fetch historical data.")
    except Exception as e:
        print(f"   -> ERROR in get_historical_data: {e}")


    # 2. Time institutional trading data fetching
    start_time = time.time()
    try:
        inst_df = fetch_institutional_trading(days=days)
        print(f"✅ fetch_institutional_trading completed in: {time.time() - start_time:.2f} seconds")
        if inst_df is None or inst_df.empty:
            print("   -> Failed to fetch institutional trading data.")
    except Exception as e:
        print(f"   -> ERROR in fetch_institutional_trading: {e}")


    # 3. Time margin trading data fetching
    start_time = time.time()
    try:
        margin_df = fetch_margin_trading(days=days)
        print(f"✅ fetch_margin_trading completed in: {time.time() - start_time:.2f} seconds")
        if margin_df is None or margin_df.empty:
            print("   -> Failed to fetch margin trading data.")
    except Exception as e:
        print(f"   -> ERROR in fetch_margin_trading: {e}")

if __name__ == "__main__":
    time_fetches()
