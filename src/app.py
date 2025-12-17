import os
import sys
import pandas as pd

# This block allows the script to be run directly without import errors
# by adding the project's root directory to the Python path.
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.data_fetcher import get_historical_data, get_top_100_stocks
from src.model import train_and_predict
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='../templates')

@app.route('/', methods=['GET'])
def index():
    """
    Renders the main page, handles stock prediction requests, and provides a list of top 100 companies.
    """
    stock_code = request.args.get('stock_code')

    # Fetch the list of top 100 companies for the dropdown menu
    top_100 = get_top_100_stocks()
    stock_list = []
    if top_100 is not None and not top_100.empty:
        stock_list = top_100[['公司代號', '公司簡稱']].to_dict('records')

    # If no stock code is provided, just render the main page
    if not stock_code:
        return render_template('index.html', stock_list=stock_list)

    # If a stock code is provided, run the prediction
    try:
        print(f"Fetching historical data for {stock_code}...")
        history_df = get_historical_data([stock_code], period="1y")

        if history_df is None or history_df.empty:
            error_msg = f"Could not find historical data for {stock_code}. Please check the stock symbol."
            return render_template('index.html', error=error_msg, stock_list=stock_list, stock_code=stock_code)

        print(f"Training model and making prediction for {stock_code}...")
        forecast_df, latest_features = train_and_predict(history_df, forecast_days=30)

        if forecast_df is None:
            error_msg = f"An error occurred while predicting for {stock_code}. There may be insufficient data."
            return render_template('index.html', error=error_msg, stock_list=stock_list, stock_code=stock_code)

        # Prepare data for the chart (last 90 days of history + 30 days of forecast)
        history_df_chart = history_df.tail(90)
        historical_data = history_df_chart.to_dict('records')
        forecast_data = forecast_df.to_dict('records')

        # Format dates
        for item in historical_data:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')
        for item in forecast_data:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')

        # Get the company name for display
        stock_name = ""
        if stock_list:
            match = next((item for item in stock_list if item['公司代號'] == stock_code), None)
            if match:
                stock_name = match['公司簡稱']

        print("Prediction complete, rendering results page.")
        return render_template('index.html',
                               stock_code=stock_code,
                               stock_name=stock_name,
                               historical_data=historical_data,
                               forecast_data=forecast_data,
                               stock_list=stock_list,
                               features=latest_features)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        error_msg = f"An unexpected error occurred: {str(e)}"
        return render_template('index.html', error=error_msg, stock_list=stock_list, stock_code=stock_code)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
