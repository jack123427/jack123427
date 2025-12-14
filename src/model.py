import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

def create_features(df):
    """
    從時間序列資料中建立更豐富的特徵工程。
    """
    df = df.copy()

    # 移动平均线
    df['MA7'] = df['ClosingPrice'].rolling(window=7).mean()
    df['MA21'] = df['ClosingPrice'].rolling(window=21).mean()
    df['MA_diff'] = df['MA7'] - df['MA21']

    # RSI (相对强弱指数) - 增加數值穩定性
    delta = df['ClosingPrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10) # 加上一個極小值以避免除以零
    df['RSI'] = 100 - (100 / (1 + rs))

    # 建立延遲特徵
    for i in range(1, 6):
        df[f'lag_{i}'] = df['ClosingPrice'].shift(i)

    return df

def train_and_predict(historical_data: pd.DataFrame, forecast_days: int = 30):
    """
    使用 XGBoost 模型訓練並預測未來 N 天的股價走勢。
    """
    if historical_data is None or len(historical_data) < 30: # 需要更多資料來計算特徵
        return None

    try:
        data = historical_data.sort_values('Date').set_index('Date')

        # 建立特徵並用 bfill 處理 NaN 值以保留資料點
        features_df = create_features(data)
        features_df.fillna(method='bfill', inplace=True)
        features_df.dropna(inplace=True) # 確保在 bfill 後仍然沒有 NaN

        FEATURES = ['MA7', 'MA21', 'MA_diff', 'RSI'] + [f'lag_{i}' for i in range(1, 6)]
        TARGET = 'ClosingPrice'

        X = features_df[FEATURES]
        y = features_df[TARGET]

        if X.empty or y.empty:
            print("After feature creation and cleaning, the dataset is empty. Cannot train model.")
            return None

        # 訓練模型
        reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
                               objective='reg:squarederror',
                               eval_metric='rmse')
        reg.fit(X, y, eval_set=[(X, y)], verbose=False)

        # === 迭代預測未來 ===
        predictions = []

        # 建立一个 DataFrame 来储存历史和未来的价格，用于动态计算特徵
        price_series = data['ClosingPrice'].copy()

        for i in range(forecast_days):
            # 取得最新的价格序列以建立特徵
            current_series_df = pd.DataFrame({'ClosingPrice': price_series})

            # 建立最新的特徵
            latest_features_df = create_features(current_series_df)
            last_features = latest_features_df[FEATURES].iloc[-1:]

            # 預測下一天
            next_pred = reg.predict(last_features)[0]
            predictions.append(next_pred)

            # 将预测结果加入价格序列，用于下一次迴圈
            next_date = price_series.index[-1] + pd.Timedelta(days=1)
            price_series.loc[next_date] = next_pred

        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({'Date': future_dates, 'PredictedPrice': predictions})
        return forecast_df

    except Exception as e:
        print(f"模型訓練或預測時發生錯誤: {e}")
        return None

def backtest_model(historical_data: pd.DataFrame, test_days: int = 10):
    """
    對 XGBoost 模型進行回測。
    """
    if historical_data is None or len(historical_data) < (test_days + 30):
        return None

    try:
        data = historical_data.sort_values('Date').set_index('Date')

        # 建立特徵並用 bfill 處理 NaN 值
        features_df = create_features(data)
        features_df.fillna(method='bfill', inplace=True)
        features_df.dropna(inplace=True)

        FEATURES = ['MA7', 'MA21', 'MA_diff', 'RSI'] + [f'lag_{i}' for i in range(1, 6)]
        TARGET = 'ClosingPrice'

        X = features_df[FEATURES]
        y = features_df[TARGET]

        if X.empty or y.empty or len(X) < test_days:
            print("After feature creation, the dataset is too small to backtest. Skipping.")
            return None

        # 分割資料
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_days, shuffle=False)

        # 訓練模型
        reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
                               objective='reg:squarederror',
                               eval_metric='rmse')
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False)

        # 預測
        predictions = reg.predict(X_test)

        # 評估
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        print(f"回測完成 - MAPE: {mape:.2f}%")

        return mape

    except Exception as e:
        print(f"模型回測時發生錯誤: {e}")
        return None

import numpy as np

if __name__ == '__main__':
    dates = pd.to_datetime(pd.date_range(start="2024-01-01", periods=100))
    # 建立一个更有趋势性的假资料
    prices = pd.Series([100 + i*0.5 + (i//10)*5 - (i//20)*3 + 5*np.sin(i/7) for i in range(100)])
    sample_history = pd.DataFrame({'Date': dates, 'ClosingPrice': prices})

    print("--- 測試 XGBoost 模型預測 ---")
    forecast_result = train_and_predict(sample_history, forecast_days=30)
    if forecast_result is not None:
        print("未來 30 天的預測結果:")
        print(forecast_result.to_string())

    print("\n" + "="*50 + "\n")

    print("--- 測試 XGBoost 模型回測 ---")
    backtest_model(sample_history, test_days=10)
