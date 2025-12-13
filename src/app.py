from flask import Flask, request, jsonify
from src.data_fetcher import get_historical_data_for_stocks
from src.model import train_and_predict

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    stock_code = request.args.get('stock_code')
    if not stock_code:
        return jsonify({"error": "請提供股票代碼 (stock_code)。"}), 400

    try:
        # 1. 獲取歷史資料
        print(f"正在為 {stock_code} 獲取歷史資料...")
        history = get_historical_data_for_stocks([stock_code], days=100)
        if history is None:
            return jsonify({"error": f"找不到 {stock_code} 的歷史資料。"}), 404

        # 2. 訓練模型並預測
        print(f"正在為 {stock_code} 進行模型訓練與預測...")
        forecast = train_and_predict(history, forecast_days=10)
        if forecast is None:
            return jsonify({"error": f"為 {stock_code} 進行預測時發生錯誤。"}), 500

        # 3. 回傳結果
        print(f"預測完成，回傳結果。")
        result = forecast.to_dict('records')
        for item in result:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"處理請求時發生未預期的錯誤: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
