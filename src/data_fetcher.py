import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import time

def get_listed_companies():
    """
    Fetches the list of publicly traded companies from the TWSE API.
    """
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        df = df[pd.to_numeric(df['公司代號'], errors='coerce').notna()]
        df['公司代號'] = df['公司代號'].astype(str)
        df['已發行普通股數或TDR原股發行股數'] = pd.to_numeric(df['已發行普通股數或TDR原股發行股數'], errors='coerce')
        return df
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch company list from TWSE API: {e}")
        return None

def get_daily_stock_data(date_str: str):
    """
    Fetches daily stock data for a given date from the TWSE API.
    """
    url = f"https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL?response=json&date={date_str}"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df = df[df['Code'].str.match(r'^\d{4}$')].copy()
        df.rename(columns={'Code': '公司代號'}, inplace=True)
        price_cols = ['OpeningPrice', 'HighestPrice', 'LowestPrice', 'ClosingPrice']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch daily stock data for {date_str} from TWSE API: {e}")
        return None

def calculate_top_100_market_cap(companies_df, prices_df):
    if companies_df is None or prices_df is None or prices_df.empty:
        return None
    merged_df = pd.merge(
        prices_df[['公司代號', 'ClosingPrice']],
        companies_df[['公司代號', '公司簡稱', '產業別', '已發行普通股數或TDR原股發行股數']],
        on='公司代號',
        how='inner'
    )
    merged_df.dropna(subset=['ClosingPrice', '已發行普通股數或TDR原股發行股數'], inplace=True)
    merged_df['市值'] = merged_df['ClosingPrice'] * merged_df['已發行普通股數或TDR原股發行股數']
    return merged_df.sort_values(by='市值', ascending=False).head(100).reset_index(drop=True)

def get_top_100_stocks():
    companies_df = get_listed_companies()
    if companies_df is None:
        print("Cannot calculate top 100 stocks without company list.")
        return None
    for i in range(5):
        date_to_try = datetime.today() - timedelta(days=i)
        date_str = date_to_try.strftime('%Y%m%d')
        latest_prices_df = get_daily_stock_data(date_str)
        if latest_prices_df is not None and not latest_prices_df.empty:
            top_100_df = calculate_top_100_market_cap(companies_df, latest_prices_df)
            if top_100_df is not None and not top_100_df.empty:
                return top_100_df
    print("Could not find valid price data in the last 5 days to calculate market cap.")
    return None

def get_historical_data(stock_codes, period="1y"):
    """
    Fetches historical data using the robust yf.Ticker().history() method,
    trying both .TW and .TWO suffixes.
    """
    if not stock_codes:
        return pd.DataFrame()

    all_data = []
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 1.1) # 1 year + buffer

    for code in stock_codes:
        data = pd.DataFrame()
        try:
            # First, try the .TW suffix for the main exchange
            ticker_tw = yf.Ticker(f"{code}.TW")
            data = ticker_tw.history(start=start_date, end=end_date, interval='1d')

            # If that returns no data, try the .TWO suffix for the OTC market
            if data.empty:
                ticker_two = yf.Ticker(f"{code}.TWO")
                data = ticker_two.history(start=start_date, end=end_date, interval='1d')

            if data.empty:
                continue

            # Reset index to make 'Date' a column and remove timezone
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)


            # Add the stock code to the DataFrame
            data['公司代號'] = code
            all_data.append(data)

        except Exception as e:
            print(f"An error occurred while fetching data for {code}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data)

    # Standardize column names and filter out days with no volume
    df = df[df['Volume'] > 0].copy()
    df.rename(columns={
        'Open': 'OpeningPrice', 'High': 'HighestPrice', 'Low': 'LowestPrice',
        'Close': 'ClosingPrice', 'Volume': 'TradeVolume'
    }, inplace=True)

    required_cols = ['Date', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'ClosingPrice', 'TradeVolume', '公司代號']

    # Ensure all required columns are present
    if not all(col in df.columns for col in required_cols):
        # Return only the columns that are actually present
        present_cols = [col for col in required_cols if col in df.columns]
        return df[present_cols]

    return df[required_cols]

def fetch_institutional_trading(days=365):
    """Fetches institutional trading data for the last N days."""
    all_data = []
    today = datetime.today()
    for i in range(days):
        date = today - timedelta(days=i)
        date_str = date.strftime('%Y%m%d')
        url = f"https://www.twse.com.tw/fund/T86?response=json&date={date_str}&selectType=ALL"
        try:
            # TWSE requires a specific user-agent
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            content = response.json()
            if content.get('stat') != 'OK' or 'data' not in content:
                continue

            df = pd.DataFrame(content['data'], columns=content['fields'])
            df['date'] = date
            all_data.append(df)
            time.sleep(0.1) # Be respectful to the API
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Failed to fetch institutional data for {date_str}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    # Standardize column names
    final_df.rename(columns={'證券代號': '公司代號', '外陸資買賣超股數(不含外資自營商)': 'Foreign_Net_Buy_Sell',
                             '投信買賣超股數': 'Investment_Trust_Net_Buy_Sell',
                             '自營商買賣超股數': 'Dealer_Net_Buy_Sell',
                             '三大法人買賣超股數': 'Total_Net_Buy_Sell'}, inplace=True)

    # Convert numbers to numeric, removing commas
    for col in ['Foreign_Net_Buy_Sell', 'Investment_Trust_Net_Buy_Sell', 'Dealer_Net_Buy_Sell', 'Total_Net_Buy_Sell']:
        final_df[col] = pd.to_numeric(final_df[col].str.replace(',', ''), errors='coerce')

    return final_df[['date', '公司代號', 'Foreign_Net_Buy_Sell', 'Investment_Trust_Net_Buy_Sell', 'Dealer_Net_Buy_Sell', 'Total_Net_Buy_Sell']]

def fetch_margin_trading(days=365):
    """Fetches margin trading data for the last N days."""
    all_data = []
    today = datetime.today()
    for i in range(days):
        date = today - timedelta(days=i)
        date_str = date.strftime('%Y%m%d')
        url = f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={date_str}&selectType=ALL"
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            content = response.json()
            if content.get('stat') != 'OK' or 'data' not in content:
                continue

            df = pd.DataFrame(content['data'], columns=content['fields'])
            df['date'] = date
            all_data.append(df)
            time.sleep(0.1)
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Failed to fetch margin data for {date_str}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.rename(columns={'股票代號': '公司代號', '融資買進': 'Margin_Buy', '融資賣出': 'Margin_Sell',
                             '融資餘額': 'Margin_Balance', '融券賣出': 'Short_Sell', '融券買進': 'Short_Cover',
                             '融券餘額': 'Short_Balance'}, inplace=True)

    cols_to_convert = ['Margin_Buy', 'Margin_Sell', 'Margin_Balance', 'Short_Sell', 'Short_Cover', 'Short_Balance']
    for col in cols_to_convert:
        final_df[col] = pd.to_numeric(final_df[col].str.replace(',', ''), errors='coerce')

    return final_df[['date', '公司代號'] + cols_to_convert]
