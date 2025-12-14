import unittest
import pandas as pd
from datetime import datetime
import sys
import os

# 將 src 目錄加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import train_and_predict, backtest_model

class TestModel(unittest.TestCase):

    def setUp(self):
        # 建立一個通用的假歷史資料，長度增加到 50 以滿足特徵計算的需求
        dates = pd.to_datetime(pd.date_range(start="2024-01-01", periods=50))
        prices = pd.Series([100 + i for i in range(50)])
        self.sample_history = pd.DataFrame({
            'Date': dates,
            'ClosingPrice': prices,
            '公司代號': '2330' # 新增公司代號
        })

    def test_train_and_predict(self):
        forecast_df, latest_features = train_and_predict(self.sample_history, forecast_days=5)
        self.assertIsNotNone(forecast_df)
        self.assertIsNotNone(latest_features)
        self.assertEqual(len(forecast_df), 5)
        self.assertIn('PredictedPrice', forecast_df.columns)

    def test_backtest_model(self):
        mape = backtest_model(self.sample_history, test_days=5)
        self.assertIsNotNone(mape)
        self.assertIsInstance(mape, float)

if __name__ == '__main__':
    unittest.main()
