# save_json.py
import json
from bvmt_forecasting_module import predict_stock, predict_volume, predict_liquidity

print("="*60)
print("SAVING FORECAST RESULTS TO JSON")
print("="*60)

# Configuration
symbol = "BIAT"
model = "prophet"

print(f"Processing {symbol} with {model} model...")

try:
    # Get all predictions
    price_forecast = predict_stock(symbol, model_type=model, forecast_days=5)
    volume_forecast = predict_volume(symbol, model_type=model, forecast_days=5)
    liquidity_analysis = predict_liquidity(symbol)
    
    print("✓ All predictions generated successfully")
    
    # 1. Save complete forecast
    with open('price_forecast.json', 'w', encoding='utf-8') as f:
        json.dump(price_forecast, f, indent=2, ensure_ascii=False)
    print("✓ Saved: price_forecast.json")
    
    # 2. Save volume forecast
    with open('volume_forecast.json', 'w', encoding='utf-8') as f:
        json.dump(volume_forecast, f, indent=2, ensure_ascii=False)
    print("✓ Saved: volume_forecast.json")
    
    # 3. Save liquidity analysis
    with open('liquidity_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(liquidity_analysis, f, indent=2, ensure_ascii=False)
    print("✓ Saved: liquidity_analysis.json")
    
    # 4. Create and save summary
    summary = {
        "symbol": symbol,
        "model": model,
        "timestamp": price_forecast['forecast_date'],
        "current_price": price_forecast['last_actual_price'],
        "next_day_forecast": {
            "date": price_forecast['forecasts'][0]['date'],
            "price": price_forecast['forecasts'][0]['predicted_close'],
            "change_percent": round(((price_forecast['forecasts'][0]['predicted_close'] - price_forecast['last_actual_price']) / price_forecast['last_actual_price']) * 100, 2),
            "confidence_interval": price_forecast['forecasts'][0]['confidence_interval_95']
        },
        "liquidity": {
            "classification": liquidity_analysis['classification'],
            "score": round(liquidity_analysis['current_liquidity_score'], 2),
            "high_probability": f"{liquidity_analysis['high_liquidity_probability']*100:.1f}%",
            "volume_ratio": round(liquidity_analysis['volume_ratio'], 2)
        },
        "recommendation": "BUY" if price_forecast['forecasts'][0]['predicted_close'] > price_forecast['last_actual_price'] else "HOLD"
    }
    
    with open('forecast_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("✓ Saved: forecast_summary.json")
    
    # 5. Save complete results
    complete_results = {
        "module": "BVMT Forecasting Module 1",
        "symbol": symbol,
        "price_forecast": price_forecast,
        "volume_forecast": volume_forecast,
        "liquidity_analysis": liquidity_analysis,
        "summary": summary
    }
    
    with open('module1_complete_results.json', 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)
    print("✓ Saved: module1_complete_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Current Price: {price_forecast['last_actual_price']:.2f} TND")
    print(f"Next Day Forecast: {price_forecast['forecasts'][0]['predicted_close']:.2f} TND")
    print(f"Expected Change: {((price_forecast['forecasts'][0]['predicted_close'] - price_forecast['last_actual_price']) / price_forecast['last_actual_price']) * 100:.2f}%")
    print(f"Confidence Interval: {price_forecast['forecasts'][0]['confidence_interval_95']}")
    print(f"Liquidity: {liquidity_analysis['classification']}")
    print(f"High Liquidity Probability: {liquidity_analysis['high_liquidity_probability']*100:.1f}%")
    print(f"Volume Ratio: {liquidity_analysis['volume_ratio']:.2f}x")
    
    print("\n✅ All JSON files created successfully!")
    print("="*60)
    
except Exception as e:
    print(f"✗ Error: {str(e)}")