import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

def create_features(df):
    """
    從時間序列資料中建立特徵。
    """
    df = df.copy()
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    return df

def train_and_predict(historical_data: pd.DataFrame, forecast_days: int = 30):
    """
    使用 XGBoost 模型訓練並預測未來 N 天的股價走勢。
    """
    if historical_data is None or len(historical_data) < 20:
        return None

    try:
        data = historical_data.sort_values('Date').set_index('Date')

        # 建立時間特徵
        features_df = create_features(data.reset_index())
        features_df.set_index('Date', inplace=True)

        # 建立延遲特徵
        for i in range(1, 6):
            features_df[f'lag_{i}'] = features_df['ClosingPrice'].shift(i)

        features_df.dropna(inplace=True)

        FEATURES = ['dayofweek', 'month', 'year', 'dayofyear'] + [f'lag_{i}' for i in range(1, 6)]
        TARGET = 'ClosingPrice'

        X = features_df[FEATURES]
        y = features_df[TARGET]

        # 訓練模型
        reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
                               objective='reg:squarederror',
                               eval_metric='rmse')
        reg.fit(X, y, eval_set=[(X, y)], verbose=False)

        # === 迭代預測未來 ===
        predictions = []

        # 獲取最後一筆真實資料的特徵和價格
        last_features = X.iloc[-1:].copy()

        for _ in range(forecast_days):
            # 預測下一天
            next_pred = reg.predict(last_features[FEATURES])[0]
            predictions.append(next_pred)

            # 建立下一天的日期
            next_date = last_features.index[0] + pd.Timedelta(days=1)

            # 建立新的特徵 DataFrame，用於下一次預測
            new_features = pd.DataFrame(index=[next_date])

            # 1. 產生新的時間特徵
            new_features = create_features(new_features.reset_index().rename(columns={'index': 'Date'}))
            new_features.set_index('Date', inplace=True)

            # 2. 滾動更新 lag 特徵
            new_features['lag_1'] = next_pred
            for i in range(2, 6):
                new_features[f'lag_{i}'] = last_features[f'lag_{i-1}'].values[0]

            # 更新 last_features 以供下一次迴圈使用
            last_features = new_features.copy()

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
    if historical_data is None or len(historical_data) < (test_days + 20):
        return None

    try:
        data = historical_data.sort_values('Date').set_index('Date')

        # 建立特徵
        features_df = create_features(data.reset_index())
        features_df.set_index('Date', inplace=True)
        for i in range(1, 6):
            features_df[f'lag_{i}'] = features_df['ClosingPrice'].shift(i)
        features_df.dropna(inplace=True)

        FEATURES = ['dayofweek', 'month', 'year', 'dayofyear'] + [f'lag_{i}' for i in range(1, 6)]
        TARGET = 'ClosingPrice'

        X = features_df[FEATURES]
        y = features_df[TARGET]

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

if __name__ == '__main__':
    dates = pd.to_datetime(pd.date_range(start="2024-01-01", periods=100))
    prices = pd.Series([100 + i + (i//10)*3 for i in range(100)])
    sample_history = pd.DataFrame({'Date': dates, 'ClosingPrice': prices})

    print("--- 測試 XGBoost 模型預測 ---")
    forecast_result = train_and_predict(sample_history, forecast_days=30)
    if forecast_result is not None:
        print("未來 30 天的預測結果:")
        print(forecast_result.to_string())

    print("\n" + "="*50 + "\n")

    print("--- 測試 XGBoost 模型回測 ---")
    backtest_model(sample_history, test_days=10)
