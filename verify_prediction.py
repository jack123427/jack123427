
import pandas as pd
from src.data_fetcher import get_historical_data
from src.model import train_and_predict
import time

def verify():
    """
    Directly tests the train_and_predict function to verify the fix
    for the flat prediction issue, bypassing the web interface.
    """
    stock_code = '2330'
    print(f"--- Verifying prediction for stock: {stock_code} ---")
    start_time = time.time()

    try:
        # 1. Fetch historical data
        print("Fetching historical data...")
        history_df = get_historical_data([stock_code], period="1y")
        print(f"Historical data fetched. Shape: {history_df.shape if history_df is not None else 'None'}")
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

        if history_df is None or history_df.empty:
            print("Error: Could not fetch historical data.")
            return

        # 2. Run the prediction logic
        print("Training model and making prediction...")
        forecast_df, _ = train_and_predict(history_df, forecast_days=30)
        print(f"Prediction complete. Forecast shape: {forecast_df.shape if forecast_df is not None else 'None'}")
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")


        if forecast_df is None or forecast_df.empty:
            print("Error: Prediction failed to return a dataframe.")
            return

        # 3. Check the results
        print("\n--- Prediction Results (first 5 days) ---")
        print(forecast_df.head())
        print("-------------------------------------------\n")

        # Check if the prediction is flat
        # Using a tolerance for floating point comparison
        is_flat = forecast_df['Predicted_Close'].nunique(dropna=True) <= 1

        if is_flat:
            print("🔥🔥🔥 VERIFICATION FAILED: The prediction is still flat.")
        else:
            print("✅✅✅ VERIFICATION SUCCESS: The prediction is varied.")

    except Exception as e:
        print(f"An error occurred during verification: {e}")
    finally:
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    verify()
