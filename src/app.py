from flask import Flask, request, render_template
from .data_fetcher import get_historical_data, get_top_100_stocks
from .model import train_and_predict
import pandas as pd

app = Flask(__name__, template_folder='../templates')

@app.route('/', methods=['GET'])
def index():
    """
    渲染主頁面，處理股票預測請求，並提供市值百大公司列表。
    """
    stock_code = request.args.get('stock_code')

    # 取得百大公司列表用於下拉選單
    top_100 = get_top_100_stocks()
    stock_list = []
    if top_100 is not None and not top_100.empty:
        stock_list = top_100[['公司代號', '公司簡稱']].to_dict('records')

    # 如果沒有輸入股票代號，只顯示主頁面
    if not stock_code:
        return render_template('index.html', stock_list=stock_list)

    # 如果有股票代號，則執行預測
    try:
        print(f"正在為 {stock_code} 獲取歷史資料...")
        # 獲取約一年的歷史資料用於特徵工程
        history_df = get_historical_data([stock_code], period="1y")

        if history_df is None or history_df.empty:
            error_msg = f"找不到 {stock_code} 的歷史資料。請確認股票代號是否正確。"
            return render_template('index.html', error=error_msg, stock_list=stock_list, stock_code=stock_code)

        print(f"正在為 {stock_code} 進行模型訓練與預測...")
        forecast_df, latest_features = train_and_predict(history_df, forecast_days=30)

        if forecast_df is None:
            error_msg = f"為 {stock_code} 進行預測時發生錯誤。可能是資料不足或模型無法收斂。"
            return render_template('index.html', error=error_msg, stock_list=stock_list, stock_code=stock_code)

        # 準備圖表資料 (最近90天歷史 + 30天預測)
        history_df_chart = history_df.tail(90)
        historical_data = history_df_chart.to_dict('records')
        forecast_data = forecast_df.to_dict('records')

        # 格式化日期
        for item in historical_data:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')
        for item in forecast_data:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')

        # 取得公司名稱用於顯示
        stock_name = ""
        if stock_list:
            match = next((item for item in stock_list if item['公司代號'] == stock_code), None)
            if match:
                stock_name = match['公司簡稱']

        print("預測完成，渲染結果頁面。")
        return render_template('index.html',
                               stock_code=stock_code,
                               stock_name=stock_name,
                               historical_data=historical_data,
                               forecast_data=forecast_data,
                               stock_list=stock_list,
                               features=latest_features)

    except Exception as e:
        print(f"處理請求時發生未預期的錯誤: {e}")
        error_msg = f"處理您的請求時發生未預期的錯誤: {str(e)}"
        return render_template('index.html', error=error_msg, stock_list=stock_list, stock_code=stock_code)

if __name__ == '__main__':
    # 使用 debug=True 以便於開發
    app.run(host='0.0.0.0', port=8080, debug=True)
