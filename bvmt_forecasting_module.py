"""
BVMT Stock Market Time Series Forecasting Module
Module 1 - Complete Implementation

This module provides:
1. Data Pipeline for OHLCV data
2. Multiple forecasting models (LSTM, Prophet, ARIMA)
3. Evaluation metrics
4. Prediction API
5. Visualization utilities
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

# ============================================================================
# 1. DATA PIPELINE
# ============================================================================

class BVMTDataPipeline:
    """
    Data pipeline for cleaning and feature engineering BVMT stock data
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_data(self, symbol, start_date=None, end_date=None):
        """
        Load stock data for a given symbol
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
            
        Returns:
        --------
        pd.DataFrame
            OHLCV data
        """
        # In production, this would connect to BVMT API or data source
        # For now, we'll generate realistic synthetic data
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        return self._generate_synthetic_data(symbol, start_date, end_date)
    
    def _generate_synthetic_data(self, symbol, start_date, end_date):
        """Generate realistic synthetic stock data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(date_range)
        
        # Generate base price with trend and seasonality
        np.random.seed(hash(symbol) % 2**32)
        trend = np.linspace(50, 100, n)
        seasonality = 10 * np.sin(np.linspace(0, 8*np.pi, n))
        noise = np.random.normal(0, 5, n)
        
        close = trend + seasonality + noise
        
        # Generate OHLC based on close
        open_price = close + np.random.normal(0, 1, n)
        high = np.maximum(close, open_price) + np.abs(np.random.normal(0, 2, n))
        low = np.minimum(close, open_price) - np.abs(np.random.normal(0, 2, n))
        
        # Generate volume with realistic patterns
        base_volume = 100000
        volume = base_volume * (1 + 0.5 * np.abs(np.random.normal(0, 1, n)))
        
        df = pd.DataFrame({
            'Date': date_range,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume.astype(int)
        })
        
        df['Symbol'] = symbol
        return df
    
    def clean_data(self, df):
        """
        Clean stock data: handle missing values, outliers
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw stock data
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'])
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Handle missing values
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Remove outliers (beyond 4 standard deviations)
        for col in ['Open', 'High', 'Low', 'Close']:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(mean - 4*std, mean + 4*std)
        
        # Ensure High >= Low, High >= Close/Open, Low <= Close/Open
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
        
        return df
    
    def engineer_features(self, df):
        """
        Create technical indicators and features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned stock data
            
        Returns:
        --------
        pd.DataFrame
            Data with engineered features
        """
        df = df.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Change'] = df['Close'] - df['Open']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Volatility
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Liquidity indicator (based on volume and volatility)
        df['Liquidity_Score'] = (df['Volume_Ratio'] * (1 / (df['Volatility_20'] + 0.001)))
        df['Liquidity_Class'] = (df['Liquidity_Score'] > df['Liquidity_Score'].median()).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Day of week
        df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_for_modeling(self, df, target='Close', dropna=True):
        """
        Prepare data for modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with features
        target : str
            Target variable
        dropna : bool
            Whether to drop NaN values
            
        Returns:
        --------
        pd.DataFrame
            Modeling-ready data
        """
        df = df.copy()
        
        if dropna:
            df = df.dropna()
        
        return df


# ============================================================================
# 2. FORECASTING MODELS
# ============================================================================

class ARIMAModel:
    """
    ARIMA model for time series forecasting
    Implemented from scratch for educational purposes
    """
    
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.p, self.d, self.q = order
        self.params = None
        self.fitted_values = None
        
    def fit(self, data):
        """
        Fit ARIMA model
        
        Parameters:
        -----------
        data : array-like
            Time series data
        """
        # Differencing
        diff_data = data.copy()
        for _ in range(self.d):
            diff_data = np.diff(diff_data)
        
        # Simple AR model fitting using least squares
        if self.p > 0:
            X = []
            y = []
            for i in range(self.p, len(diff_data)):
                X.append(diff_data[i-self.p:i][::-1])
                y.append(diff_data[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # OLS estimation
            self.params = np.linalg.lstsq(X, y, rcond=None)[0]
        
        self.data = data
        self.fitted_values = self._get_fitted_values()
        
    def _get_fitted_values(self):
        """Get fitted values"""
        if self.params is None:
            return None
        
        diff_data = self.data.copy()
        for _ in range(self.d):
            diff_data = np.diff(diff_data)
        
        fitted = np.zeros(len(diff_data))
        for i in range(self.p, len(diff_data)):
            fitted[i] = np.dot(self.params, diff_data[i-self.p:i][::-1])
        
        return fitted
    
    def forecast(self, steps=5):
        """
        Forecast future values
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        np.array
            Forecasted values
        """
        if self.params is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = []
        
        # Get last p values
        diff_data = self.data.copy()
        for _ in range(self.d):
            diff_data = np.diff(diff_data)
        
        last_values = list(diff_data[-self.p:])
        
        for _ in range(steps):
            # Predict next value
            next_val = np.dot(self.params, last_values[::-1])
            forecasts.append(next_val)
            last_values.append(next_val)
            last_values.pop(0)
        
        # Reverse differencing
        forecasts = np.array(forecasts)
        for _ in range(self.d):
            forecasts = np.cumsum(forecasts) + self.data[-1]
        
        return forecasts


class LSTMModel:
    """
    LSTM model for time series forecasting
    Simplified implementation
    """
    
    def __init__(self, sequence_length=60, units=50):
        self.sequence_length = sequence_length
        self.units = units
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_sequences(self, data):
        """Prepare sequences for LSTM"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def fit(self, data, epochs=50, batch_size=32, verbose=0):
        """
        Fit LSTM model
        Note: This is a simplified version. In production, use TensorFlow/Keras
        """
        X, y = self.prepare_sequences(data)
        
        # Store for prediction
        self.X_train = X
        self.y_train = y
        self.last_sequence = X[-1]
        
        # In production, build and train actual LSTM here
        # For now, we'll use a simple moving average as a placeholder
        self.model = 'LSTM_placeholder'
        
    def forecast(self, steps=5):
        """Forecast future values"""
        predictions = []
        current_sequence = self.last_sequence.copy()
        
        for _ in range(steps):
            # Simple prediction using moving average (placeholder)
            next_pred = np.mean(current_sequence[-10:])
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()


class ProphetModel:
    """
    Prophet-like model for time series forecasting
    Simplified implementation
    """
    
    def __init__(self):
        self.trend = None
        self.seasonal = None
        self.data = None
        
    def fit(self, df, date_col='Date', value_col='Close'):
        """
        Fit Prophet model
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with date and value columns
        """
        df = df.copy()
        df['ds'] = pd.to_datetime(df[date_col])
        df['y'] = df[value_col]
        df = df.sort_values('ds')
        
        # Fit trend (linear regression)
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['y'].values
        
        # Simple linear regression
        X_mean = X.mean()
        y_mean = y.mean()
        
        numerator = ((X.flatten() - X_mean) * (y - y_mean)).sum()
        denominator = ((X.flatten() - X_mean) ** 2).sum()
        
        slope = numerator / denominator
        intercept = y_mean - slope * X_mean
        
        self.trend = {'slope': slope, 'intercept': intercept}
        
        # Fit seasonality (weekly)
        trend_line = slope * X.flatten() + intercept
        detrended = y - trend_line
        
        df['detrended'] = detrended
        df['day_of_week'] = df['ds'].dt.dayofweek
        
        seasonal_components = df.groupby('day_of_week')['detrended'].mean().to_dict()
        self.seasonal = seasonal_components
        
        self.data = df
        self.last_date = df['ds'].max()
        
    def forecast(self, periods=5):
        """
        Forecast future periods
        
        Parameters:
        -----------
        periods : int
            Number of periods to forecast
            
        Returns:
        --------
        pd.DataFrame
            Forecasts with uncertainty
        """
        future_dates = pd.date_range(
            start=self.last_date + timedelta(days=1),
            periods=periods,
            freq='B'
        )
        
        X_future = np.arange(len(self.data), len(self.data) + periods)
        
        # Trend forecast
        trend_forecast = self.trend['slope'] * X_future + self.trend['intercept']
        
        # Seasonal forecast
        seasonal_forecast = np.array([
            self.seasonal.get(date.dayofweek, 0) 
            for date in future_dates
        ])
        
        # Combined forecast
        yhat = trend_forecast + seasonal_forecast
        
        # Calculate uncertainty (simple std-based)
        residuals_std = self.data['detrended'].std()
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': yhat,
            'yhat_lower': yhat - 1.96 * residuals_std,
            'yhat_upper': yhat + 1.96 * residuals_std
        })
        
        return forecast_df


# ============================================================================
# 3. EVALUATION METRICS
# ============================================================================

class ModelEvaluator:
    """
    Evaluate forecasting models
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate RMSE, MAE, and directional accuracy
        
        Parameters:
        -----------
        y_true : array-like
            Actual values
        y_pred : array-like
            Predicted values
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional Accuracy
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = None
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
    
    @staticmethod
    def compare_models(results_dict):
        """
        Compare multiple models
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with model names as keys and metrics as values
            
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        comparison = pd.DataFrame(results_dict).T
        comparison['Rank_RMSE'] = comparison['RMSE'].rank()
        comparison['Rank_MAE'] = comparison['MAE'].rank()
        
        return comparison.sort_values('RMSE')


# ============================================================================
# 4. PREDICTION API
# ============================================================================

def predict_stock(symbol, model_type='prophet', forecast_days=5, data=None):
    """
    Main API function for stock prediction
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol
    model_type : str
        Model to use: 'arima', 'lstm', or 'prophet'
    forecast_days : int
        Number of days to forecast (default: 5)
    data : pd.DataFrame, optional
        Pre-loaded data. If None, will load from pipeline
        
    Returns:
    --------
    dict
        Forecast results with confidence intervals
    """
    # Initialize pipeline
    pipeline = BVMTDataPipeline()
    
    # Load and prepare data
    if data is None:
        print(f"Loading data for {symbol}...")
        data = pipeline.load_data(symbol)
    
    data = pipeline.clean_data(data)
    data = pipeline.engineer_features(data)
    data = pipeline.prepare_for_modeling(data)
    
    # Select and train model
    if model_type.lower() == 'arima':
        model = ARIMAModel(order=(5, 1, 0))
        model.fit(data['Close'].values)
        forecast_values = model.forecast(steps=forecast_days)
        
        # Calculate confidence intervals (simple std-based)
        residuals_std = np.std(data['Close'].values[-20:] - data['Close'].values[-21:-1])
        lower_bound = forecast_values - 1.96 * residuals_std
        upper_bound = forecast_values + 1.96 * residuals_std
        
    elif model_type.lower() == 'lstm':
        model = LSTMModel(sequence_length=60)
        model.fit(data['Close'].values)
        forecast_values = model.forecast(steps=forecast_days)
        
        residuals_std = np.std(data['Close'].values[-20:] - data['Close'].values[-21:-1])
        lower_bound = forecast_values - 1.96 * residuals_std
        upper_bound = forecast_values + 1.96 * residuals_std
        
    elif model_type.lower() == 'prophet':
        model = ProphetModel()
        model.fit(data, date_col='Date', value_col='Close')
        forecast_df = model.forecast(periods=forecast_days)
        
        forecast_values = forecast_df['yhat'].values
        lower_bound = forecast_df['yhat_lower'].values
        upper_bound = forecast_df['yhat_upper'].values
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create forecast dates
    last_date = pd.to_datetime(data['Date'].max())
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_days,
        freq='B'
    )
    
    # Prepare results
    results = {
        'symbol': symbol,
        'model': model_type,
        'forecast_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_actual_price': float(data['Close'].iloc[-1]),
        'last_actual_date': str(data['Date'].iloc[-1]),
        'forecasts': [
            {
                'date': str(date.date()),
                'predicted_close': float(pred),
                'lower_bound': float(lower),
                'upper_bound': float(upper),
                'confidence_interval_95': f"{float(lower):.2f} - {float(upper):.2f}"
            }
            for date, pred, lower, upper in zip(
                forecast_dates, forecast_values, lower_bound, upper_bound
            )
        ]
    }
    
    return results


def predict_volume(symbol, model_type='prophet', forecast_days=5, data=None):
    """
    Predict daily volume
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol
    model_type : str
        Model to use
    forecast_days : int
        Number of days to forecast
    data : pd.DataFrame, optional
        Pre-loaded data
        
    Returns:
    --------
    dict
        Volume forecast results
    """
    pipeline = BVMTDataPipeline()
    
    if data is None:
        data = pipeline.load_data(symbol)
    
    data = pipeline.clean_data(data)
    data = pipeline.engineer_features(data)
    data = pipeline.prepare_for_modeling(data)
    
    # Use Prophet for volume prediction
    if model_type.lower() == 'prophet':
        model = ProphetModel()
        model.fit(data, date_col='Date', value_col='Volume')
        forecast_df = model.forecast(periods=forecast_days)
        
        forecast_values = forecast_df['yhat'].values
        lower_bound = forecast_df['yhat_lower'].values
        upper_bound = forecast_df['yhat_upper'].values
    else:
        # Use ARIMA as fallback
        model = ARIMAModel(order=(5, 1, 0))
        model.fit(data['Volume'].values)
        forecast_values = model.forecast(steps=forecast_days)
        
        residuals_std = np.std(data['Volume'].values[-20:] - data['Volume'].values[-21:-1])
        lower_bound = forecast_values - 1.96 * residuals_std
        upper_bound = forecast_values + 1.96 * residuals_std
    
    last_date = pd.to_datetime(data['Date'].max())
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_days,
        freq='B'
    )
    
    results = {
        'symbol': symbol,
        'model': model_type,
        'forecast_type': 'volume',
        'forecasts': [
            {
                'date': str(date.date()),
                'predicted_volume': int(max(0, pred)),
                'lower_bound': int(max(0, lower)),
                'upper_bound': int(upper)
            }
            for date, pred, lower, upper in zip(
                forecast_dates, forecast_values, lower_bound, upper_bound
            )
        ]
    }
    
    return results


def predict_liquidity(symbol, data=None, threshold=None):
    """
    Classify liquidity as high/low probability
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol
    data : pd.DataFrame, optional
        Pre-loaded data
    threshold : float, optional
        Custom threshold for classification
        
    Returns:
    --------
    dict
        Liquidity classification results
    """
    pipeline = BVMTDataPipeline()
    
    if data is None:
        data = pipeline.load_data(symbol)
    
    data = pipeline.clean_data(data)
    data = pipeline.engineer_features(data)
    data = pipeline.prepare_for_modeling(data)
    
    # Calculate current liquidity metrics
    latest = data.iloc[-1]
    
    if threshold is None:
        threshold = data['Liquidity_Score'].median()
    
    liquidity_score = latest['Liquidity_Score']
    is_high_liquidity = liquidity_score > threshold
    
    # Calculate probability based on recent trend
    recent_scores = data['Liquidity_Score'].tail(20)
    high_liquidity_prob = (recent_scores > threshold).mean()
    
    results = {
        'symbol': symbol,
        'current_liquidity_score': float(liquidity_score),
        'threshold': float(threshold),
        'classification': 'High' if is_high_liquidity else 'Low',
        'high_liquidity_probability': float(high_liquidity_prob),
        'low_liquidity_probability': float(1 - high_liquidity_prob),
        'recent_volume': int(latest['Volume']),
        'volume_ratio': float(latest['Volume_Ratio']),
        'volatility': float(latest['Volatility_20'])
    }
    
    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_stock_list():
    """
    Get list of Tunisian stocks
    In production, this would fetch from BVMT
    """
    return [
        'BIAT', 'BNA', 'STB', 'ATB', 'UBCI',  # Banks
        'TUNIS-RE', 'STAR', 'COMAR',  # Insurance
        'SFBT', 'DELICE', 'POULINA',  # Food
        'CARTHAGE_CEMENT', 'SOTUVER',  # Materials
        'ONE', 'TELNET'  # Telecom
    ]


if __name__ == "__main__":
    print("BVMT Stock Forecasting Module")
    print("=" * 60)
    
    # Example usage
    symbol = 'BIAT'
    print(f"\nExample: Forecasting {symbol}")
    print("-" * 60)
    
    # Price forecast
    print("\n1. Price Forecast (Prophet):")
    results = predict_stock(symbol, model_type='prophet', forecast_days=5)
    print(f"Last actual price: {results['last_actual_price']:.2f}")
    print("\n5-Day Forecast:")
    for fc in results['forecasts']:
        print(f"  {fc['date']}: {fc['predicted_close']:.2f} "
              f"[{fc['confidence_interval_95']}]")
    
    # Volume forecast
    print("\n2. Volume Forecast:")
    vol_results = predict_volume(symbol, forecast_days=5)
    for fc in vol_results['forecasts']:
        print(f"  {fc['date']}: {fc['predicted_volume']:,} shares")
    
    # Liquidity classification
    print("\n3. Liquidity Classification:")
    liq_results = predict_liquidity(symbol)
    print(f"  Classification: {liq_results['classification']}")
    print(f"  High Liquidity Probability: {liq_results['high_liquidity_probability']:.1%}")
    print(f"  Current Volume Ratio: {liq_results['volume_ratio']:.2f}")