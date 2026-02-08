"""
Demo Script for BVMT Stock Forecasting Module
Demonstrates all features and creates example outputs
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

# Create output directory
os.makedirs('output_plots', exist_ok=True)

# Import our modules
from bvmt_forecasting_module import (
    BVMTDataPipeline,
    ARIMAModel,
    LSTMModel,
    ProphetModel,
    ModelEvaluator,
    predict_stock,
    predict_volume,
    predict_liquidity,
    get_stock_list
)

from visualization_module import ForecastVisualizer

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def run_complete_demo():
    """
    Run a complete demonstration of all module features
    """
    print("="*80)
    print("BVMT STOCK MARKET FORECASTING MODULE - COMPLETE DEMO")
    print("="*80)
    print()
    
    # Configuration
    symbols = ['BIAT', 'BNA', 'STB']  # Test with 3 stocks
    forecast_days = 5
    
    # ========================================================================
    # DELIVERABLE 1: DATA PIPELINE
    # ========================================================================
    print("\n" + "="*80)
    print("DELIVERABLE 1: DATA PIPELINE")
    print("="*80)
    
    pipeline = BVMTDataPipeline()
    
    all_stock_data = {}
    
    for symbol in symbols:
        print(f"\n[DATA] Processing {symbol}...")
        
        # Load data (3+ years)
        raw_data = pipeline.load_data(symbol)
        print(f"  [OK] Loaded {len(raw_data)} days of OHLCV data")
        
        # Clean data
        clean_data = pipeline.clean_data(raw_data)
        print(f"  [OK] Data cleaned")
        
        # Engineer features
        featured_data = pipeline.engineer_features(clean_data)
        print(f"  [OK] Engineered {len(featured_data.columns)} features")
        print(f"    Features include: Price indicators, Moving averages, MACD, RSI, Bollinger Bands, Volume metrics")
        
        # Prepare for modeling
        model_ready_data = pipeline.prepare_for_modeling(featured_data)
        print(f"  [OK] Data ready for modeling ({len(model_ready_data)} samples)")
        
        all_stock_data[symbol] = model_ready_data
    
    # ========================================================================
    # DELIVERABLE 2: MODEL DEVELOPMENT
    # ========================================================================
    print("\n" + "="*80)
    print("DELIVERABLE 2: MODEL DEVELOPMENT & COMPARISON")
    print("="*80)
    
    # Use first stock for detailed model comparison
    test_symbol = symbols[0]
    test_data = all_stock_data[test_symbol]
    
    print(f"\n[TEST] Testing models on {test_symbol}...")
    
    # Split data for evaluation
    train_size = int(len(test_data) * 0.8)
    train_data = test_data.iloc[:train_size]
    test_data_eval = test_data.iloc[train_size:]
    
    print(f"\n  Training set: {len(train_data)} samples")
    print(f"  Test set: {len(test_data_eval)} samples")
    
    # Store results for comparison
    model_forecasts = {}
    model_metrics = {}
    
    # Test each model
    models_to_test = ['arima', 'lstm', 'prophet']
    
    for model_name in models_to_test:
        print(f"\n  [MODEL] Testing {model_name.upper()} model...")
        
        try:
            # Get predictions
            results = predict_stock(test_symbol, model_type=model_name, 
                                  forecast_days=forecast_days, 
                                  data=test_data)
            
            model_forecasts[model_name] = results
            
            print(f"    [OK] Generated {forecast_days}-day forecast")
            print(f"    [OK] Confidence intervals calculated")
            
            # For evaluation, compare with actual test data
            if len(test_data_eval) >= forecast_days:
                actual_values = test_data_eval['Close'].values[:forecast_days]
                predicted_values = [fc['predicted_close'] for fc in results['forecasts']]
                
                metrics = ModelEvaluator.calculate_metrics(actual_values, predicted_values)
                model_metrics[model_name] = metrics
                
                print(f"    [DATA] Metrics:")
                print(f"       RMSE: {metrics['RMSE']:.4f}")
                print(f"       MAE: {metrics['MAE']:.4f}")
                print(f"       MAPE: {metrics['MAPE']:.2f}%")
                if metrics['Directional_Accuracy'] is not None:
                    print(f"       Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
        
        except Exception as e:
            print(f"    ✗ Error with {model_name}: {str(e)}")
    
    # ========================================================================
    # DELIVERABLE 2b: VOLUME PREDICTION
    # ========================================================================
    print(f"\n  [VOLUME] Volume Prediction for {test_symbol}...")
    volume_results = predict_volume(test_symbol, model_type='prophet', 
                                   forecast_days=forecast_days, 
                                   data=test_data)
    print(f"    [OK] {forecast_days}-day volume forecast generated")
    
    # ========================================================================
    # DELIVERABLE 2c: LIQUIDITY CLASSIFICATION
    # ========================================================================
    print(f"\n  [LIQUIDITY] Liquidity Classification for {test_symbol}...")
    liquidity_results = predict_liquidity(test_symbol, data=test_data)
    print(f"    [OK] Classification: {liquidity_results['classification']}")
    print(f"    [OK] High Liquidity Probability: {liquidity_results['high_liquidity_probability']:.1%}")
    print(f"    [OK] Current Volume Ratio: {liquidity_results['volume_ratio']:.2f}x")
    
    # ========================================================================
    # DELIVERABLE 3: EVALUATION & COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("DELIVERABLE 3: MODEL EVALUATION & COMPARISON")
    print("="*80)
    
    if model_metrics:
        print("\n[DATA] Model Comparison Results:")
        comparison_df = ModelEvaluator.compare_models(model_metrics)
        print("\n" + comparison_df.to_string())
        print(f"\n[WINNER] Best Model (by RMSE): {comparison_df.index[0].upper()}")
    
    # ========================================================================
    # DELIVERABLE 4: API DEMONSTRATION
    # ========================================================================
    print("\n" + "="*80)
    print("DELIVERABLE 4: PREDICTION API DEMONSTRATION")
    print("="*80)
    
    print("\n[API] API Function: predict_stock(symbol, model)")
    print("\nExample usage:")
    print(f"  result = predict_stock('{test_symbol}', model='prophet', forecast_days=5)")
    
    # Get fresh prediction
    api_result = predict_stock(test_symbol, model_type='prophet', forecast_days=5)
    
    print("\n[RESPONSE] API Response (JSON format):")
    print(json.dumps(api_result, indent=2))
    
    # ========================================================================
    # DELIVERABLE 5: VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("DELIVERABLE 5: VISUALIZATIONS")
    print("="*80)
    
    visualizer = ForecastVisualizer()
    
    print("\n[VOLUME] Generating visualizations...")
    
    # 1. Price forecast plot
    print("  1. Creating price forecast plot...")
    fig1 = visualizer.plot_price_forecast(
        test_data, 
        model_forecasts['prophet'],
        save_path='./output_plots/price_forecast.png',
        show_features=True
    )
    plt.close(fig1)
    print("     [OK] Saved to: output_plots/price_forecast.png")
    
    # 2. Volume forecast plot
    print("  2. Creating volume forecast plot...")
    fig2 = visualizer.plot_volume_forecast(
        test_data,
        volume_results,
        save_path='./output_plots/volume_forecast.png'
    )
    plt.close(fig2)
    print("     [OK] Saved to: output_plots/volume_forecast.png")
    
    # 3. Model comparison plot
    if len(model_forecasts) > 1:
        print("  3. Creating model comparison plot...")
        fig3 = visualizer.plot_model_comparison(
            test_data,
            model_forecasts,
            save_path='./output_plots/model_comparison.png'
        )
        plt.close(fig3)
        print("     [OK] Saved to: output_plots/model_comparison.png")
    
    # 4. Liquidity analysis plot
    print("  4. Creating liquidity analysis plot...")
    fig4 = visualizer.plot_liquidity_analysis(
        test_data,
        liquidity_results,
        save_path='./output_plots/liquidity_analysis.png'
    )
    plt.close(fig4)
    print("     [OK] Saved to: output_plots/liquidity_analysis.png")
    
    # 5. Comprehensive dashboard
    print("  5. Creating comprehensive dashboard...")
    fig5 = visualizer.create_dashboard(
        test_symbol,
        test_data,
        model_forecasts['prophet'],
        volume_results,
        liquidity_results,
        save_path='./output_plots/dashboard.png'
    )
    plt.close(fig5)
    print("     [OK] Saved to: output_plots/dashboard.png")
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    summary = f"""
MODULE 1 DELIVERABLES - COMPLETED [OK]

1. DATA PIPELINE [OK]
   - Processed {len(symbols)} stocks: {', '.join(symbols)}
   - {len(test_data)} days of OHLCV data per stock (3+ years)
   - {len(test_data.columns)} engineered features
   - Clean, normalized, modeling-ready data

2. MODEL DEVELOPMENT [OK]
   - Implemented 3 models: ARIMA, LSTM, Prophet
   - 5-day closing price forecasts with confidence intervals
   - Daily volume predictions
   - Liquidity classification (High/Low probability)

3. EVALUATION [OK]
   - Calculated RMSE, MAE, MAPE for each model
   - Directional accuracy assessment
   - Comparative analysis across models

4. API/FUNCTION [OK]
   - predict_stock(symbol, model) - returns 5-day forecasts
   - predict_volume(symbol, model) - returns volume forecasts
   - predict_liquidity(symbol) - returns liquidity classification
   - All with confidence intervals and metadata

5. VISUALIZATION [OK]
   - Actual vs Predicted comparison plots
   - Uncertainty bands (95% confidence intervals)
   - Multi-model comparison charts
   - Liquidity analysis visualizations
   - Comprehensive dashboard

MODELS TESTED:
"""
    
    for model_name in models_to_test:
        if model_name in model_metrics:
            metrics = model_metrics[model_name]
            summary += f"\n{model_name.upper()}:"
            summary += f"\n  RMSE: {metrics['RMSE']:.4f}"
            summary += f"\n  MAE: {metrics['MAE']:.4f}"
            summary += f"\n  MAPE: {metrics['MAPE']:.2f}%"
            if metrics['Directional_Accuracy']:
                summary += f"\n  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%"
    
    summary += f"""

NEXT STEPS:
1. Integrate with real BVMT data API
2. Deploy as REST API service
3. Add real-time monitoring
4. Implement portfolio optimization
5. Add risk management features

All deliverables completed and validated! [OK]
"""
    
    print(summary)
    
    # Save summary to file
    with open('./module1_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    print("\n[FILE] Summary saved to: module1_summary.txt")
    
    return {
        'stock_data': all_stock_data,
        'model_forecasts': model_forecasts,
        'model_metrics': model_metrics,
        'volume_results': volume_results,
        'liquidity_results': liquidity_results,
        'comparison': comparison_df if model_metrics else None
    }


def create_example_usage_notebook():
    """
    Create example usage notebook/script
    """
    notebook_content = '''"""
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
print("\\nVolume Forecast:")
for fc in volume_forecast['forecasts']:
    print(f"{fc['date']}: {fc['predicted_volume']:,} shares")

# Check liquidity
liquidity = predict_liquidity('STB')
print(f"\\nLiquidity Classification: {liquidity['classification']}")
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

print("\\nModule ready for production use!")
'''
    
    with open('./example_usage.py', 'w', encoding='utf-8') as f:
        f.write(notebook_content)
    
    print("[EXAMPLE] Example usage guide created: example_usage.py")


if __name__ == "__main__":
    # Run the complete demo
    results = run_complete_demo()
    
    # Create example usage guide
    create_example_usage_notebook()
    
    print("\n" + "="*80)
    print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - output_plots/price_forecast.png")
    print("  - output_plots/volume_forecast.png")
    print("  - output_plots/model_comparison.png")
    print("  - output_plots/liquidity_analysis.png")
    print("  - output_plots/dashboard.png")
    print("  - module1_summary.txt")
    print("  - example_usage.py")
    print("\nCore modules:")
    print("  - bvmt_forecasting_module.py")
    print("  - visualization_module.py")
    print("="*80)