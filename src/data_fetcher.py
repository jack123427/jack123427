import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def get_listed_companies():
    """
    從 TWSE API 取得所有台灣上市公司的基本資料。
    """
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        # 過濾非數值和無效的公司代號
        df = df[pd.to_numeric(df['公司代號'], errors='coerce').notna()]
        df['公司代號'] = df['公司代號'].astype(str)
        df['已發行普通股數或TDR原股發行股數'] = pd.to_numeric(df['已發行普通股數或TDR原股發行股數'], errors='coerce')
        return df
    except requests.exceptions.RequestException as e:
        print(f"無法從 TWSE API 取得上市公司列表: {e}")
        return None

def get_daily_stock_data(date_str: str):
    """
    從 TWSE API 取得指定日期的所有股票每日交易資料。
    """
    url = f"https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL?response=json&date={date_str}"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # 過濾掉 '證券代號' 不是4位數字的行
        df = df[df['Code'].str.match(r'^\d{4}$')].copy()
        df.rename(columns={'Code': '公司代號'}, inplace=True)
        price_cols = ['OpeningPrice', 'HighestPrice', 'LowestPrice', 'ClosingPrice']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except requests.exceptions.RequestException as e:
        print(f"無法從 TWSE API 取得 {date_str} 的股價資料: {e}")
        return None

def calculate_top_100_market_cap(companies_df, prices_df):
    """
    根據公司資料和當日股價計算市值百大公司。
    """
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

def get_top_100_stocks():
    """
    取得最新市值百大公司的列表。
    這是一個輔助函數，整合了獲取公司列表、最新股價和計算市值的過程。
    """
    companies_df = get_listed_companies()
    if companies_df is None:
        print("無法取得上市公司列表，無法計算百大公司。")
        return None

    # 從今天開始往前找，直到找到有交易資料的一天
    for i in range(5):
        date_to_try = datetime.today() - timedelta(days=i)
        date_str = date_to_try.strftime('%Y%m%d')
        latest_prices_df = get_daily_stock_data(date_str)
        if latest_prices_df is not None and not latest_prices_df.empty:
            print(f"成功取得 {date_str} 的股價資料。")
            top_100_df = calculate_top_100_market_cap(companies_df, latest_prices_df)
            if top_100_df is not None and not top_100_df.empty:
                return top_100_df
            else:
                 print(f"在 {date_str} 的資料無法計算市值排名。")
        else:
            print(f"無法取得 {date_str} 的股價資料，嘗試前一天...")

    print("在過去5天內都無法找到有效的股價資料來計算市值排名。")
    return None


def get_historical_data(stock_codes, period="1y"):
    """
    使用 yfinance 取得一個或多個股票代號的歷史股價資料。
    """
    if not stock_codes:
        return pd.DataFrame()

    tickers = [f"{code}.TW" for code in stock_codes]
    try:
        data = yf.download(tickers, period=period, progress=False)

        if data.empty:
            print(f"yfinance 未能為代號 {stock_codes} 返回任何資料。")
            return pd.DataFrame()

        # 如果只有一個 ticker，yfinance 返回的 DataFrame 結構不同，需要標準化
        if len(stock_codes) == 1:
            data['公司代號'] = stock_codes[0]
            df = data.reset_index()
        else:
            # 對於多個 ticker，需要將 multi-level columns 轉換為單層
            data = data.stack(level=0).rename_axis(['Date', 'Attributes']).reset_index(level=1)
            df = data.reset_index()
            # 從 'Attributes' 中提取公司代號
            df['公司代號'] = df['Attributes'].apply(lambda x: x.replace('.TW', ''))

        # 過濾掉沒有交易量的日子並重命名欄位以保持一致性
        df = df[df['Volume'] > 0].copy()
        df.rename(columns={
            'Open': 'OpeningPrice',
            'High': 'HighestPrice',
            'Low': 'LowestPrice',
            'Close': 'ClosingPrice',
            'Volume': 'TradeVolume'
        }, inplace=True)

        # 確保必要的欄位存在
        required_cols = ['Date', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'ClosingPrice', 'TradeVolume', '公司代號']
        for col in required_cols:
            if col not in df.columns:
                print(f"警告: yfinance 返回的資料缺少欄位 '{col}'")
                return pd.DataFrame()

        return df[required_cols]

    except Exception as e:
        print(f"使用 yfinance 獲取歷史資料時出錯: {e}")
        return pd.DataFrame()
