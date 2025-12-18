import unittest
import pandas as pd
from datetime import datetime
import sys
import os

# 將專案根目錄加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import train_and_predict, backtest_model, create_features

class TestModel(unittest.TestCase):

    def setUp(self):
        """Set up a more realistic sample DataFrame and ensure the models directory exists."""
        dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=50, freq='D'))
        prices = [100 + i + (i % 5) for i in range(50)]
        self.sample_history = pd.DataFrame({
            'Date': dates,
            'OpeningPrice': [p - 1 for p in prices],
            'HighestPrice': [p + 1 for p in prices],
            'LowestPrice': [p - 2 for p in prices],
            'ClosingPrice': prices,
            'TradingVolume': [10000 + i * 100 for i in range(50)],
            '公司代號': '2330'
        })
        os.makedirs("models", exist_ok=True)

    def tearDown(self):
        """Clean up any cached model and feature files created during the test."""
        model_path = "models/2330.joblib"
        features_path = "models/2330_features.joblib"
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(features_path):
            os.remove(features_path)

    def test_create_features(self):
        """Test that feature creation works and handles data correctly."""
        features_df = create_features(self.sample_history, stock_code='2330')
        self.assertIn('MA7', features_df.columns)
        self.assertIn('RSI', features_df.columns)
        # Technical indicators like rolling averages will create NaNs at the start.
        # The main logic handles this, so we just check that the columns were created.
        self.assertTrue(features_df['MA7'].isnull().any())
        self.assertTrue(features_df['RSI'].isnull().any())

    def test_train_and_predict(self):
        """Test the full training and prediction pipeline, ensuring no caching interference."""
        # Ensure a clean slate by removing any previously cached models.
        self.tearDown()

        forecast_df, latest_features = train_and_predict(self.sample_history, forecast_days=5)

        self.assertIsNotNone(forecast_df, "Forecast DataFrame should not be None")
        self.assertIsNotNone(latest_features, "Latest features dictionary should not be None")
        self.assertEqual(len(forecast_df), 5, "Forecast should contain 5 days of data")
        self.assertIn('PredictedPrice', forecast_df.columns)

    def test_backtest_model(self):
        """Test that the backtest function returns a plausible MAPE score."""
        mape = backtest_model(self.sample_history, test_days=5)
        self.assertIsNotNone(mape)
        self.assertIsInstance(mape, float)
        self.assertGreaterEqual(mape, 0.0)

if __name__ == '__main__':
    unittest.main()
