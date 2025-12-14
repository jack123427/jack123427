from flask import Flask, request, render_template, jsonify
from src.data_fetcher import get_historical_data_for_stocks
from src.model import train_and_predict
import pandas as pd

app = Flask(__name__, template_folder='../templates')

@app.route('/predict', methods=['GET'])
def predict():
    stock_code = request.args.get('stock_code')

    # 如果沒有提供股票代碼，只顯示頁面，並傳遞空值
    if not stock_code:
        return render_template('index.html',
                               stock_code=None,
                               historical_data=None,
                               forecast_data=None)

    try:
        # 1. 獲取歷史資料
        print(f"正在為 {stock_code} 獲取歷史資料...")
        history_df = get_historical_data_for_stocks([stock_code], days=100)
        if history_df is None or history_df.empty:
            return render_template('index.html', error=f"找不到 {stock_code} 的歷史資料。")

        # 2. 訓練模型並預測
        print(f"正在為 {stock_code} 進行模型訓練與預測...")
        forecast_df = train_and_predict(history_df, forecast_days=30)
        if forecast_df is None:
            return render_template('index.html', error=f"為 {stock_code} 進行預測時發生錯誤。")

        # 3. 準備要傳遞給前端的資料
        # 將 DataFrame 轉換為字典列表
        historical_data = history_df.to_dict('records')
        forecast_data = forecast_df.to_dict('records')

        # 格式化日期以便 JavaScript 處理
        for item in historical_data:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')
        for item in forecast_data:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')

        print("預測完成，渲染結果頁面。")
        return render_template('index.html',
                               stock_code=stock_code,
                               historical_data=historical_data,
                               forecast_data=forecast_data)

    except Exception as e:
        print(f"處理請求時發生未預期的錯誤: {e}")
        return render_template('index.html', error=f"處理請求時發生未預期的錯誤: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
