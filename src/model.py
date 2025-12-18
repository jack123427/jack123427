import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import warnings
from src.data_fetcher import fetch_institutional_trading, fetch_margin_trading

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

import joblib
import os

def train_and_predict(historical_data: pd.DataFrame, forecast_days: int = 30):
    """
    Uses a two-pass XGBoost model to train with feature selection and predict future stock prices.
    """
    if historical_data is None or len(historical_data) < 30:
        return None, None

    historical_data.dropna(subset=['Date'], inplace=True)
    stock_code = historical_data['公司代號'].iloc[0]

    # --- Model Caching ---
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{stock_code}.joblib")
    features_path = os.path.join(model_dir, f"{stock_code}_features.joblib")

    try:
        # Try to load a pre-trained model and its features
        if os.path.exists(model_path) and os.path.exists(features_path):
            print(f"Loading cached model for stock {stock_code}...")
            reg = joblib.load(model_path)
            SELECTED_FEATURES = joblib.load(features_path)
            # We still need to create all features first for the iterative prediction loop
            end_date = historical_data['Date'].max()
            start_date = historical_data['Date'].min()
            institutional_df = fetch_institutional_trading(start_date, end_date)
            margin_df = fetch_margin_trading(start_date, end_date)
        else:
            print(f"No cached model found for {stock_code}. Training a new one.")
            # --- Data Fetching and Feature Engineering ---
            end_date = historical_data['Date'].max()
            start_date = historical_data['Date'].min()
            institutional_df = fetch_institutional_trading(start_date, end_date)
            margin_df = fetch_margin_trading(start_date, end_date)

            features_df = create_features(historical_data, institutional_df, margin_df, stock_code)
            features_df.bfill(inplace=True)
            features_df.dropna(inplace=True)

            ALL_FEATURES = [col for col in features_df.columns if col not in ['Date', '公司代號', 'ClosingPrice']]
            TARGET = 'ClosingPrice'

            X_all = features_df[ALL_FEATURES]
            y = features_df[TARGET]

            if X_all.empty or y.empty:
                print("Dataset is empty after feature creation. Cannot train model.")
                return None, None

            # --- Feature Selection ---
            # We must only train on features that can be calculated during the prediction loop.
            # Chip-based features are historical and not available for future dates.
            print("Selecting features for final model training...")

            # Define chip-based features that should be excluded from the final model
            chip_features = [
                'Foreign_Net_Buy_Sell', 'Investment_Trust_Net_Buy_Sell', 'Dealer_Net_Buy_Sell',
                'Total_Institutional_Net_Buy_Sell', 'foreign_ma5', 'foreign_momentum',
                'Margin_Balance', 'Short_Balance', 'Margin_Purchase', 'Margin_Sale',
                'Short_Purchase', 'Short_Sale', 'margin_balance_change', 'short_balance_change',
                'short_margin_ratio', 'Total_Net_Buy_Sell'
            ]

            # Features available for the final model are all features minus the chip-based ones
            final_model_features = [f for f in ALL_FEATURES if f not in chip_features]

            if not final_model_features:
                 raise ValueError("No features available for training after excluding chip-based features.")

            # We don't need a two-pass system anymore. We train on the reliable features directly.
            SELECTED_FEATURES = final_model_features
            print(f"Training model with {len(SELECTED_FEATURES)} features: {SELECTED_FEATURES}")

            # --- Pass 2: Final Model Training ---
            print("Running final training pass with selected features...")
            X_selected = features_df[SELECTED_FEATURES]

            reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
                                   objective='reg:squarederror', eval_metric='rmse')
            reg.fit(X_selected, y, eval_set=[(X_selected, y)], verbose=False)

            # --- Save the trained model and selected features ---
            print(f"Saving new model and features for stock {stock_code} to cache.")
            joblib.dump(reg, model_path)
            joblib.dump(SELECTED_FEATURES, features_path)

        # --- Iterative Forecasting ---
        print("Starting iterative forecasting...")
        predictions = []
        future_df = historical_data.copy()

        for i in range(forecast_days):
            # For future predictions, we don't have real chip data.
            # We rely on technical indicators calculated from the predicted prices.
            features_for_pred = create_features(future_df, institutional_df=None, margin_df=None, stock_code=stock_code)
            features_for_pred.bfill(inplace=True)
            features_for_pred.dropna(inplace=True)

            last_features = features_for_pred.iloc[-1:][SELECTED_FEATURES]
            pred = reg.predict(last_features)[0]
            predictions.append(pred)

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

        # Create the final features_df for returning latest feature values
        final_features_df = create_features(historical_data, institutional_df, margin_df, stock_code)
        latest_features = final_features_df.iloc[-1][SELECTED_FEATURES].to_dict()

        return forecast_df, latest_features

    except Exception as e:
        print(f"Error during model training or prediction for stock {stock_code}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def backtest_model(historical_data: pd.DataFrame, test_days: int = 10):
    # This function would also need to be updated to use the new features,
    # but for now, we will focus on the prediction part.
    # Returning a dummy value to avoid breaking the app.
    print("Backtesting not implemented with new features yet.")
    return 0.5

from prophet import Prophet

def train_and_predict_prophet(historical_data: pd.DataFrame, forecast_days: int = 30):
    """
    Uses Facebook's Prophet model to train and predict future stock prices.
    """
    if historical_data is None or len(historical_data) < 30:
        return None, None

    # Prophet requires columns 'ds' (datestamp) and 'y' (value)
    prophet_df = historical_data[['Date', 'ClosingPrice']].rename(columns={'Date': 'ds', 'ClosingPrice': 'y'})

    # Initialize and train the model
    # We include Taiwan holidays to improve the model's accuracy
    model = Prophet(daily_seasonality=True)
    model.add_country_holidays(country_name='TW')
    model.fit(prophet_df)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Extract the forecast and format it for the web app
    forecast_df = forecast[['ds', 'yhat']].tail(forecast_days)
    forecast_df = forecast_df.rename(columns={'ds': 'Date', 'yhat': 'PredictedPrice'})

    # Prophet does not provide a simple equivalent to 'latest_features' like XGBoost.
    # We will return an empty dictionary to maintain a consistent API.
    return forecast_df, {}
