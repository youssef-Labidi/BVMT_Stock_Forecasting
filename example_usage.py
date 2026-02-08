"""
Example Usage Guide for BVMT Forecasting Module

This script demonstrates how to use the forecasting module
in your own projects.
"""

from bvmt_forecasting_module import predict_stock, predict_volume, predict_liquidity
from visualization_module import ForecastVisualizer

# ============================================================================
# EXAMPLE 1: Simple Price Forecast
# ============================================================================

# Get 5-day forecast for BIAT using Prophet model
result = predict_stock('BIAT', model_type='prophet', forecast_days=5)

print("5-Day Price Forecast for BIAT:")
for forecast in result['forecasts']:
    print(f"{forecast['date']}: {forecast['predicted_close']:.2f} TND "
          f"[{forecast['confidence_interval_95']}]")

# ============================================================================
# EXAMPLE 2: Compare Multiple Models
# ============================================================================

models = ['arima', 'lstm', 'prophet']
results = {}

for model in models:
    results[model] = predict_stock('BNA', model_type=model, forecast_days=5)
    
# Visualize comparison
visualizer = ForecastVisualizer()
# (You would load historical data here)
# visualizer.plot_model_comparison(historical_data, results, save_path='comparison.png')

# ============================================================================
# EXAMPLE 3: Volume and Liquidity Analysis
# ============================================================================

# Predict volume
volume_forecast = predict_volume('STB', forecast_days=5)
print("\nVolume Forecast:")
for fc in volume_forecast['forecasts']:
    print(f"{fc['date']}: {fc['predicted_volume']:,} shares")

# Check liquidity
liquidity = predict_liquidity('STB')
print(f"\nLiquidity Classification: {liquidity['classification']}")
print(f"High Liquidity Probability: {liquidity['high_liquidity_probability']:.1%}")

# ============================================================================
# EXAMPLE 4: Batch Processing Multiple Stocks
# ============================================================================

from bvmt_forecasting_module import get_stock_list

stocks = get_stock_list()
all_forecasts = {}

for stock in stocks[:5]:  # Process first 5 stocks
    try:
        all_forecasts[stock] = predict_stock(stock, model_type='prophet')
        print(f"[OK] Processed {stock}")
    except Exception as e:
        print(f"✗ Error with {stock}: {e}")

# ============================================================================
# EXAMPLE 5: Custom Data Pipeline
# ============================================================================

from bvmt_forecasting_module import BVMTDataPipeline

pipeline = BVMTDataPipeline()

# Load your own data
# data = pd.read_csv('your_data.csv')

# Or generate synthetic data for testing
data = pipeline.load_data('TEST_STOCK')

# Clean and engineer features
data = pipeline.clean_data(data)
data = pipeline.engineer_features(data)
data = pipeline.prepare_for_modeling(data)

# Now use with any model
result = predict_stock('TEST_STOCK', model_type='prophet', data=data)

print("\nModule ready for production use!")
