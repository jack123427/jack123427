from data_fetcher import (
    get_listed_companies,
    get_daily_stock_data,
    calculate_top_100_market_cap,
    get_historical_data_for_stocks,
    get_financial_statements,
    get_daily_pe_pb,
    get_margin_trading
)
from model import train_and_predict, backtest_model
import pandas as pd
from datetime import datetime, timedelta

def main():
    """
    主要執行流程
    """
    # 1. 取得市值前一百大公司
    print("--- 步驟 1: 正在取得市值前一百大公司 ---")
    companies = get_listed_companies()
    last_trading_day_prices = get_daily_stock_data(datetime.today().strftime('%Y%m%d'))

    days_to_check = 1
    while last_trading_day_prices is None or last_trading_day_prices.empty:
        print(f"找不到 {datetime.today().strftime('%Y%m%d')} 的資料，嘗試前一天...")
        last_trading_date = datetime.today() - timedelta(days=days_to_check)
        last_trading_day_prices = get_daily_stock_data(last_trading_date.strftime('%Y%m%d'))
        days_to_check += 1
        if days_to_check > 10:
            print("錯誤：過去 10 天內都找不到交易資料。")
            return

    top_100 = calculate_top_100_market_cap(companies, last_trading_day_prices)
    if top_100 is None:
        print("錯誤：無法計算市值排名。")
        return

    print("成功取得市值前一百大公司！")

    target_stocks = top_100.head(5)
    target_stock_codes = target_stocks['公司代號'].tolist()
    print(f"將處理以下公司: {target_stock_codes}")

    # 2. 獲取特徵資料
    print("\n--- 步驟 2: 正在獲取特徵資料 ---")

    historical_prices = get_historical_data_for_stocks(target_stock_codes, days=60) # 縮短天數
    if historical_prices is None:
        print("錯誤：無法取得歷史股價資料。")
        return

    margin_trading_data = get_margin_trading(last_trading_day_prices.iloc[0]['Date'])


    # 3. 遍歷每支股票，進行訓練、預測與回測
    print("\n--- 步驟 3: 正在進行模型訓練、預測與回測 ---")
    for stock_code in target_stock_codes:
        print(f"\n===== 正在處理股票: {stock_code} =====")

        stock_history = historical_prices[historical_prices['公司代號'] == stock_code].copy()

        print(f"--- {stock_code} 模型回測 ---")
        backtest_model(stock_history, test_days=30)

        print(f"\n--- {stock_code} 未來 30 天預測 ---")
        forecast = train_and_predict(stock_history, forecast_days=30)
        if forecast is not None:
            print(forecast.to_string())

if __name__ == '__main__':
    main()
