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

def get_historical_data_for_stocks(stock_codes: list, days: int = 100, end_date: datetime = None):
    """
    取得一個股票代號列表在過去 N 個「交易日」的每日交易資料。
    """
    if end_date is None:
        end_date = datetime.today()

    unique_days_data = {}
    current_date = end_date - timedelta(days=1)
    days_collected = 0
    days_searched = 0

    # 持續回溯，直到收集到足夠天數的「獨立交易日」資料
    while days_collected < days and days_searched < (days * 3): # 最多搜尋 days * 3 的範圍
        date_str = current_date.strftime('%Y%m%d')
        daily_data = get_daily_stock_data(date_str)

        if daily_data is not None and not daily_data.empty:
            # 使用 API 回傳的日期作為唯一鍵，避免假日重複問題
            api_date_str = daily_data['Date'].iloc[0]
            if api_date_str not in unique_days_data:
                unique_days_data[api_date_str] = daily_data
                days_collected += 1

        current_date -= timedelta(days=1)
        days_searched += 1
        time.sleep(0.1)

    if not unique_days_data:
        return None

    all_data = list(unique_days_data.values())
    df = pd.concat(all_data, ignore_index=True)

    # 篩選出目標股票
    df = df[df['公司代號'].isin(stock_codes)].copy()
    if df.empty:
        return None

    # 轉換日期格式
    def convert_roc_to_ad(roc_date):
        roc_date_str = str(roc_date)
        year = int(roc_date_str[:-4]) + 1911
        month = int(roc_date_str[-4:-2])
        day = int(roc_date_str[-2:])
        return datetime(year, month, day)
    df['Date'] = df['Date'].apply(convert_roc_to_ad)

    # 為每支股票選取最新的 N 天資料
    final_df = df.groupby('公司代號').apply(lambda x: x.nlargest(days, 'Date')).reset_index(drop=True)
    final_df.sort_values(by=['公司代號', 'Date'], ascending=[True, True], inplace=True)

    return final_df

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
