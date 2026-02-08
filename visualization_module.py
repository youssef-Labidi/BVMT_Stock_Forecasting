"""
Visualization Module for BVMT Stock Forecasting
Generates comparison plots with uncertainty bands
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')


class ForecastVisualizer:
    """
    Create visualizations for stock forecasts
    """
    
    def __init__(self, figsize=(14, 8)):
        self.figsize = figsize
        
    def plot_price_forecast(self, historical_data, forecast_results, 
                           save_path=None, show_features=True):
        """
        Plot actual vs predicted prices with uncertainty bands
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical stock data
        forecast_results : dict
            Forecast results from predict_stock()
        save_path : str, optional
            Path to save the plot
        show_features : bool
            Whether to show technical indicators
        """
        fig = plt.figure(figsize=(16, 10))
        
        if show_features:
            # Create subplots
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
        else:
            ax1 = fig.add_subplot(111)
        
        # Prepare data
        hist_dates = pd.to_datetime(historical_data['Date'])
        hist_close = historical_data['Close'].values
        
        # Extract forecast data
        forecast_dates = [fc['date'] for fc in forecast_results['forecasts']]
        forecast_dates = pd.to_datetime(forecast_dates)
        forecast_values = [fc['predicted_close'] for fc in forecast_results['forecasts']]
        lower_bounds = [fc['lower_bound'] for fc in forecast_results['forecasts']]
        upper_bounds = [fc['upper_bound'] for fc in forecast_results['forecasts']]
        
        # Plot 1: Price and Forecast
        # Historical prices
        ax1.plot(hist_dates, hist_close, 'b-', linewidth=2, 
                label='Historical Close Price', alpha=0.7)
        
        # Moving averages
        if 'MA_20' in historical_data.columns:
            ax1.plot(hist_dates, historical_data['MA_20'], 'g--', 
                    linewidth=1.5, label='MA 20', alpha=0.5)
        if 'MA_50' in historical_data.columns:
            ax1.plot(hist_dates, historical_data['MA_50'], 'r--', 
                    linewidth=1.5, label='MA 50', alpha=0.5)
        
        # Forecast
        # Connect last historical point to first forecast
        connection_dates = [hist_dates.iloc[-1], forecast_dates[0]]
        connection_values = [hist_close[-1], forecast_values[0]]
        
        ax1.plot(connection_dates, connection_values, 'orange', 
                linewidth=2, linestyle='--', alpha=0.5)
        ax1.plot(forecast_dates, forecast_values, 'orange', 
                linewidth=3, label='Forecast', marker='o', markersize=8)
        
        # Uncertainty band
        ax1.fill_between(forecast_dates, lower_bounds, upper_bounds, 
                        color='orange', alpha=0.2, 
                        label='95% Confidence Interval')
        
        # Styling
        ax1.set_title(f"{forecast_results['symbol']} - Stock Price Forecast "
                     f"({forecast_results['model'].upper()} Model)", 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (TND)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add annotation for last price and first forecast
        last_price = hist_close[-1]
        first_forecast = forecast_values[0]
        change_pct = ((first_forecast - last_price) / last_price) * 100
        
        ax1.annotate(f'Last: {last_price:.2f} TND', 
                    xy=(hist_dates.iloc[-1], last_price),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    fontsize=9, fontweight='bold')
        
        ax1.annotate(f'Forecast: {first_forecast:.2f} TND\n({change_pct:+.2f}%)', 
                    xy=(forecast_dates[0], first_forecast),
                    xytext=(10, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.7),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if show_features:
            # Plot 2: Volume
            ax2.bar(hist_dates, historical_data['Volume'], 
                   color='steelblue', alpha=0.6, label='Volume')
            if 'Volume_MA_20' in historical_data.columns:
                ax2.plot(hist_dates, historical_data['Volume_MA_20'], 
                        'r-', linewidth=2, label='Volume MA 20')
            ax2.set_ylabel('Volume', fontsize=11, fontweight='bold')
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: RSI
            if 'RSI' in historical_data.columns:
                ax3.plot(hist_dates, historical_data['RSI'], 
                        'purple', linewidth=2, label='RSI')
                ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
                ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
                ax3.fill_between(hist_dates, 30, 70, alpha=0.1, color='gray')
                ax3.set_ylabel('RSI', fontsize=11, fontweight='bold')
                ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
                ax3.legend(loc='best', fontsize=9)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim([0, 100])
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig
    
    def plot_volume_forecast(self, historical_data, volume_results, save_path=None):
        """
        Plot volume forecast
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical stock data
        volume_results : dict
            Volume forecast results
        save_path : str, optional
            Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data
        hist_dates = pd.to_datetime(historical_data['Date'])
        hist_volume = historical_data['Volume'].values
        
        forecast_dates = [fc['date'] for fc in volume_results['forecasts']]
        forecast_dates = pd.to_datetime(forecast_dates)
        forecast_volumes = [fc['predicted_volume'] for fc in volume_results['forecasts']]
        
        # Plot historical volume
        ax.bar(hist_dates, hist_volume, color='steelblue', 
               alpha=0.6, label='Historical Volume', width=0.8)
        
        # Plot forecast
        ax.bar(forecast_dates, forecast_volumes, color='orange', 
               alpha=0.8, label='Forecasted Volume', width=0.8)
        
        # Add moving average
        if 'Volume_MA_20' in historical_data.columns:
            ax.plot(hist_dates, historical_data['Volume_MA_20'], 
                   'r-', linewidth=2, label='Volume MA 20')
        
        # Styling
        ax.set_title(f"{volume_results['symbol']} - Volume Forecast", 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volume (shares)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Volume plot saved to: {save_path}")
        
        return fig
    
    def plot_model_comparison(self, historical_data, model_results_dict, save_path=None):
        """
        Compare multiple models
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical stock data
        model_results_dict : dict
            Dictionary with model names as keys and forecast results as values
        save_path : str, optional
            Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Prepare historical data
        hist_dates = pd.to_datetime(historical_data['Date'])
        hist_close = historical_data['Close'].values
        
        # Plot historical data
        ax1.plot(hist_dates, hist_close, 'b-', linewidth=2, 
                label='Historical Close Price', alpha=0.7)
        
        # Define colors for different models
        colors = ['orange', 'green', 'red', 'purple', 'brown']
        
        # Plot each model's forecast
        for idx, (model_name, results) in enumerate(model_results_dict.items()):
            color = colors[idx % len(colors)]
            
            forecast_dates = pd.to_datetime([fc['date'] for fc in results['forecasts']])
            forecast_values = [fc['predicted_close'] for fc in results['forecasts']]
            lower_bounds = [fc['lower_bound'] for fc in results['forecasts']]
            upper_bounds = [fc['upper_bound'] for fc in results['forecasts']]
            
            # Plot forecast
            ax1.plot(forecast_dates, forecast_values, color=color, 
                    linewidth=2.5, label=f'{model_name.upper()} Forecast', 
                    marker='o', markersize=6)
            
            # Uncertainty band
            ax1.fill_between(forecast_dates, lower_bounds, upper_bounds, 
                            color=color, alpha=0.15)
        
        # Styling for first subplot
        ax1.set_title('Model Comparison - Price Forecasts', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (TND)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Forecast differences
        first_model_forecasts = list(model_results_dict.values())[0]['forecasts']
        forecast_dates = pd.to_datetime([fc['date'] for fc in first_model_forecasts])
        
        for idx, (model_name, results) in enumerate(model_results_dict.items()):
            color = colors[idx % len(colors)]
            forecast_values = [fc['predicted_close'] for fc in results['forecasts']]
            
            # Calculate percentage change from last historical price
            last_price = hist_close[-1]
            pct_changes = [(val - last_price) / last_price * 100 
                          for val in forecast_values]
            
            ax2.plot(forecast_dates, pct_changes, color=color, 
                    linewidth=2, label=f'{model_name.upper()}', 
                    marker='s', markersize=6)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_title('Forecast Deviation from Last Price', 
                     fontsize=16, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Change (%)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        return fig
    
    def plot_liquidity_analysis(self, historical_data, liquidity_results, save_path=None):
        """
        Visualize liquidity analysis
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical stock data
        liquidity_results : dict
            Liquidity classification results
        save_path : str, optional
            Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        hist_dates = pd.to_datetime(historical_data['Date'])
        
        # Plot 1: Liquidity Score over time
        ax1.plot(hist_dates, historical_data['Liquidity_Score'], 
                'b-', linewidth=2, label='Liquidity Score')
        ax1.axhline(y=liquidity_results['threshold'], color='r', 
                   linestyle='--', label='Threshold', linewidth=2)
        ax1.fill_between(hist_dates, liquidity_results['threshold'], 
                        historical_data['Liquidity_Score'], 
                        where=historical_data['Liquidity_Score'] >= liquidity_results['threshold'],
                        color='green', alpha=0.3, label='High Liquidity')
        ax1.fill_between(hist_dates, liquidity_results['threshold'], 
                        historical_data['Liquidity_Score'], 
                        where=historical_data['Liquidity_Score'] < liquidity_results['threshold'],
                        color='red', alpha=0.3, label='Low Liquidity')
        ax1.set_title('Liquidity Score Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Liquidity Score', fontsize=11, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volume Ratio
        ax2.plot(hist_dates, historical_data['Volume_Ratio'], 
                'purple', linewidth=2, label='Volume Ratio')
        ax2.axhline(y=1, color='orange', linestyle='--', 
                   label='Average Volume', linewidth=2)
        ax2.set_title('Volume Ratio (Volume / MA20)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volume Ratio', fontsize=11, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Volatility
        ax3.plot(hist_dates, historical_data['Volatility_20'], 
                'red', linewidth=2, label='20-day Volatility')
        ax3.set_title('Price Volatility', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Volatility', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Classification Probability
        categories = ['High Liquidity', 'Low Liquidity']
        probabilities = [
            liquidity_results['high_liquidity_probability'],
            liquidity_results['low_liquidity_probability']
        ]
        colors_pie = ['green', 'red']
        
        wedges, texts, autotexts = ax4.pie(probabilities, labels=categories, 
                                            autopct='%1.1f%%', startangle=90,
                                            colors=colors_pie, textprops={'fontsize': 12})
        ax4.set_title(f"Current Classification: {liquidity_results['classification']}", 
                     fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Liquidity analysis plot saved to: {save_path}")
        
        return fig
    
    def create_dashboard(self, symbol, historical_data, price_forecast, 
                        volume_forecast, liquidity_results, save_path=None):
        """
        Create a comprehensive dashboard
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        historical_data : pd.DataFrame
            Historical data
        price_forecast : dict
            Price forecast results
        volume_forecast : dict
            Volume forecast results
        liquidity_results : dict
            Liquidity results
        save_path : str, optional
            Path to save
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        hist_dates = pd.to_datetime(historical_data['Date'])
        
        # Main price chart (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(hist_dates, historical_data['Close'], 'b-', linewidth=2, alpha=0.7)
        
        forecast_dates = pd.to_datetime([fc['date'] for fc in price_forecast['forecasts']])
        forecast_values = [fc['predicted_close'] for fc in price_forecast['forecasts']]
        lower_bounds = [fc['lower_bound'] for fc in price_forecast['forecasts']]
        upper_bounds = [fc['upper_bound'] for fc in price_forecast['forecasts']]
        
        ax1.plot(forecast_dates, forecast_values, 'orange', linewidth=3, 
                marker='o', label='Forecast')
        ax1.fill_between(forecast_dates, lower_bounds, upper_bounds, 
                        color='orange', alpha=0.2)
        ax1.set_title(f'{symbol} - Price Forecast Dashboard', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (TND)', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Metrics summary (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        last_price = historical_data['Close'].iloc[-1]
        first_forecast = forecast_values[0]
        price_change = first_forecast - last_price
        price_change_pct = (price_change / last_price) * 100
        
        summary_text = f"""
        SUMMARY METRICS
        {'='*30}
        
        Last Price: {last_price:.2f} TND
        Next Day Forecast: {first_forecast:.2f} TND
        Expected Change: {price_change:+.2f} ({price_change_pct:+.2f}%)
        
        Liquidity: {liquidity_results['classification']}
        Volume Ratio: {liquidity_results['volume_ratio']:.2f}x
        Volatility: {liquidity_results['volatility']:.4f}
        
        Model: {price_forecast['model'].upper()}
        Forecast Date: {price_forecast['forecast_date']}
        """
        
        ax2.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        # Volume (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(hist_dates, historical_data['Volume'], color='steelblue', alpha=0.6)
        ax3.set_title('Trading Volume', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Volume', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # RSI (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'RSI' in historical_data.columns:
            ax4.plot(hist_dates, historical_data['RSI'], 'purple', linewidth=2)
            ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax4.set_title('RSI Indicator', fontsize=12, fontweight='bold')
            ax4.set_ylabel('RSI', fontsize=10)
            ax4.set_ylim([0, 100])
            ax4.grid(True, alpha=0.3)
        
        # Liquidity Score (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(hist_dates, historical_data['Liquidity_Score'], 'green', linewidth=2)
        ax5.axhline(y=liquidity_results['threshold'], color='r', linestyle='--')
        ax5.set_title('Liquidity Score', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Score', fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Volume forecast (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        vol_forecast_dates = pd.to_datetime([fc['date'] for fc in volume_forecast['forecasts']])
        vol_forecast_values = [fc['predicted_volume'] for fc in volume_forecast['forecasts']]
        ax6.bar(vol_forecast_dates, vol_forecast_values, color='orange', alpha=0.8)
        ax6.set_title('Volume Forecast', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Volume', fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Returns distribution (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        returns = historical_data['Returns'].dropna()
        ax7.hist(returns, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax7.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax7.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Daily Returns', fontsize=10)
        ax7.set_ylabel('Frequency', fontsize=10)
        ax7.grid(True, alpha=0.3)
        
        # Liquidity classification pie (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        categories = ['High', 'Low']
        probs = [liquidity_results['high_liquidity_probability'],
                liquidity_results['low_liquidity_probability']]
        colors_pie = ['green', 'red']
        ax8.pie(probs, labels=categories, autopct='%1.1f%%', 
               colors=colors_pie, startangle=90)
        ax8.set_title('Liquidity Classification', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        return fig


if __name__ == "__main__":
    # Example usage would go here
    print("Visualization module loaded successfully")