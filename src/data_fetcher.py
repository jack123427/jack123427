import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_listed_companies():
    """
    取得所有台灣上市公司的基本資料。
    """
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df['公司代號'] = df['公司代號'].astype(str)
        df['已發行普通股數或TDR原股發行股數'] = pd.to_numeric(df['已發行普通股數或TDR原股發行股數'], errors='coerce')
        return df
    except requests.exceptions.RequestException as e:
        return None

def get_daily_stock_data(date_str: str):
    """
    取得指定日期的所有股票每日交易資料。
    """
    url = f"https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL?response=json&date={date_str}"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df.rename(columns={'Code': '公司代號'}, inplace=True)
        price_cols = ['OpeningPrice', 'HighestPrice', 'LowestPrice', 'ClosingPrice']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except requests.exceptions.RequestException:
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
    top_100 = merged_df.sort_values(by='市值', ascending=False).head(100)
    return top_100.reset_index(drop=True)

def get_historical_data_for_stocks(stock_codes: list, days: int = 100):
    """
    取得一個股票代號列表在過去 N 天的每日交易資料。
    """
    all_data = []
    start_date = datetime.today() - timedelta(days=1)
    for i in range(days):
        date_to_fetch = start_date - timedelta(days=i)
        date_str = date_to_fetch.strftime('%Y%m%d')
        daily_data = get_daily_stock_data(date_str)
        if daily_data is not None and not daily_data.empty:
            filtered_data = daily_data[daily_data['公司代號'].isin(stock_codes)]
            if not filtered_data.empty:
                all_data.append(filtered_data)
        time.sleep(0.2) # 縮短 sleep 時間

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)
    def convert_roc_to_ad(roc_date):
        roc_date_str = str(roc_date)
        year = int(roc_date_str[:-4]) + 1911
        month = int(roc_date_str[-4:-2])
        day = int(roc_date_str[-2:])
        return datetime(year, month, day)
    df['Date'] = df['Date'].apply(convert_roc_to_ad)
    df.sort_values(by=['公司代號', 'Date'], ascending=[True, True], inplace=True)
    return df.reset_index(drop=True)

def get_financial_statements(year: int, season: int):
    """
    取得指定季度的綜合損益表。
    """
    url = f"https://openapi.twse.com.tw/v1/opendata/t187ap06_L_ci_{year}{season:02d}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException:
        return None

def get_daily_pe_pb(date_str: str):
    """
    取得指定日期的本益比和股價淨值比。
    """
    url = f"https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_d?response=json&date={date_str}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException:
        return None

def get_margin_trading(date_str: str):
    """
    取得指定日期的融資融券餘額。
    """
    url = f"https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN?response=json&date={date_str}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException:
        return None
