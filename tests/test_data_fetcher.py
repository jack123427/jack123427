import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import sys
import os
import requests # 修正 NameError

# 將 src 目錄加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_fetcher import get_listed_companies, get_daily_stock_data

class TestDataFetcher(unittest.TestCase):

    @patch('data_fetcher.requests.get')
    def test_get_listed_companies_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'公司代號': '2330', '公司簡稱': '台積電', '產業別': '半導體', '已發行普通股數或TDR原股發行股數': '25930380458'},
            {'公司代號': '2317', '公司簡稱': '鴻海', '產業別': '電腦及週邊設備業', '已發行普通股數或TDR原股發行股數': '13862897450'}
        ]
        mock_get.return_value = mock_response

        df = get_listed_companies()
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['公司簡稱'], '台積電')

    @patch('data_fetcher.requests.get')
    def test_get_listed_companies_api_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        df = get_listed_companies()
        self.assertIsNone(df)

    @patch('data_fetcher.requests.get')
    def test_get_daily_stock_data_success(self, mock_get):
        # 模擬 API 回應，加入所有需要的欄位
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'Code': '2330', 'Name': '台積電', 'OpeningPrice': '890.00', 'HighestPrice': '910.00', 'LowestPrice': '880.00', 'ClosingPrice': '900.00'},
            {'Code': '2317', 'Name': '鴻海', 'OpeningPrice': '190.00', 'HighestPrice': '210.00', 'LowestPrice': '180.00', 'ClosingPrice': '200.00'}
        ]
        mock_get.return_value = mock_response

        df = get_daily_stock_data('20240726')
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['公司代號'], '2330')
        self.assertEqual(df.iloc[0]['ClosingPrice'], 900.00)

if __name__ == '__main__':
    unittest.main()
