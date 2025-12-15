import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import warnings
from .data_fetcher import fetch_institutional_trading, fetch_margin_trading

warnings.filterwarnings("ignore")

def create_features(price_df, institutional_df=None, margin_df=None, stock_code=None):
    """
    Merges all data sources and creates a rich set of features.
    """
    df = price_df.copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Filter and merge institutional data
    if institutional_df is not None and not institutional_df.empty:
        inst_filtered = institutional_df[institutional_df['公司代號'] == stock_code].copy()
        inst_filtered['date'] = pd.to_datetime(inst_filtered['date']).dt.tz_localize(None)
        df = pd.merge(df, inst_filtered, left_on='Date', right_on='date', how='left')
        df.drop(columns=['date', '公司代號_y'], errors='ignore', inplace=True)
        df.rename(columns={'公司代號_x': '公司代號'}, inplace=True)


    # Filter and merge margin data
    if margin_df is not None and not margin_df.empty:
        margin_filtered = margin_df[margin_df['公司代號'] == stock_code].copy()
        margin_filtered['date'] = pd.to_datetime(margin_filtered['date']).dt.tz_localize(None)
        df = pd.merge(df, margin_filtered, left_on='Date', right_on='date', how='left')
        df.drop(columns=['date', '公司代號_y'], errors='ignore', inplace=True)
        df.rename(columns={'公司代號_x': '公司代號'}, inplace=True)

    # Fill missing values from merges
    df.ffill(inplace=True)
    df.fillna(0, inplace=True) # Fill any remaining NaNs with 0

    # Technical Indicators
    df['MA7'] = df['ClosingPrice'].rolling(window=7).mean()
    df['MA21'] = df['ClosingPrice'].rolling(window=21).mean()
    df['MA_diff'] = df['MA7'] - df['MA21']

    delta = df['ClosingPrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    for i in range(1, 6):
        df[f'lag_{i}'] = df['ClosingPrice'].shift(i)

    # Chip-based Indicators
    if 'Foreign_Net_Buy_Sell' in df.columns:
        df['foreign_ma5'] = df['Foreign_Net_Buy_Sell'].rolling(5).mean()
        df['foreign_momentum'] = df['Foreign_Net_Buy_Sell'].diff(5)

    if 'Margin_Balance' in df.columns and 'Short_Balance' in df.columns:
        df['margin_balance_change'] = df['Margin_Balance'].pct_change()
        df['short_balance_change'] = df['Short_Balance'].pct_change()
        df['short_margin_ratio'] = df['Short_Balance'] / (df['Margin_Balance'] + 1e-10)

    return df

def train_and_predict(historical_data: pd.DataFrame, forecast_days: int = 30):
    """
    Uses XGBoost model to train and predict future stock prices.
    """
    if historical_data is None or len(historical_data) < 30:
        return None, None

    historical_data.dropna(subset=['Date'], inplace=True)
    stock_code = historical_data['公司代號'].iloc[0]

    try:
        # Define date range for fetching chip data
        end_date = historical_data['Date'].max()
        start_date = historical_data['Date'].min()

        # Fetch new data
        print(f"Fetching institutional and margin data from {start_date} to {end_date}...")
        institutional_df = fetch_institutional_trading(start_date, end_date)
        margin_df = fetch_margin_trading(start_date, end_date)

        # Filter for the specific stock
        # Create features
        features_df = create_features(historical_data, institutional_df, margin_df, stock_code)
        features_df.bfill(inplace=True)
        features_df.dropna(inplace=True)

        FEATURES = [col for col in features_df.columns if col not in ['Date', '公司代號', 'ClosingPrice']]
        TARGET = 'ClosingPrice'

        X = features_df[FEATURES]
        y = features_df[TARGET]

        if X.empty or y.empty:
            print("After feature creation, the dataset is empty. Cannot train model.")
            return None, None

        reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
                               objective='reg:squarederror',
                               eval_metric='rmse')
        reg.fit(X, y, eval_set=[(X, y)], verbose=False)

        # Iterative forecasting
        print("Starting iterative forecasting...")
        predictions = []
        # Start with the full historical data for feature creation
        future_df = historical_data.copy()

        for i in range(forecast_days):
            # 1. Create features for the current dataset
            features_for_pred = create_features(future_df, institutional_df, margin_df, stock_code)
            features_for_pred.bfill(inplace=True)
            features_for_pred.dropna(inplace=True)

            # 2. Predict the next step using the last row of features
            last_features = features_for_pred.iloc[-1:][FEATURES]
            pred = reg.predict(last_features)[0]
            predictions.append(pred)

            # 3. Create a new row for the next day and append it
            last_date = future_df['Date'].iloc[-1]
            new_row = pd.DataFrame({
                'Date': [last_date + pd.Timedelta(days=1)],
                'ClosingPrice': [pred],
                '公司代號': [stock_code]
            })
            future_df = pd.concat([future_df, new_row], ignore_index=True)
        print("Iterative forecasting complete.")


        future_dates = pd.date_range(start=historical_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({'Date': future_dates, 'PredictedPrice': predictions})

        # Return the latest features as well
        latest_features = features_df.iloc[-1][FEATURES].to_dict()

        return forecast_df, latest_features

    except Exception as e:
        print(f"Error during model training or prediction: {e}")
        return None, None

def backtest_model(historical_data: pd.DataFrame, test_days: int = 10):
    # This function would also need to be updated to use the new features,
    # but for now, we will focus on the prediction part.
    # Returning a dummy value to avoid breaking the app.
    print("Backtesting not implemented with new features yet.")
    return 0.5
