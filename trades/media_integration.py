import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class GDELTStockIntegrator:
    """
    Integrates GDELT media coverage data with stock price data
    Creates features for volume and returns prediction
    """
    
    def __init__(self, stock_file, gdelt_file, ticker, output_dir='trades/data/input/media_integrated'):
        """
        Args:
            stock_file: Path to stock CSV (Date, Open, High, Low, Close, Adj Close, Volume)
            gdelt_file: Path to GDELT CSV (Date, Ticker, ArticleCount, Tone, Polarity, WordCount)
            ticker: Stock ticker to process (e.g., 'AMSC')
            output_dir: Directory to save processed data
        """
        self.stock_file = stock_file
        self.gdelt_file = gdelt_file
        self.ticker = ticker
        self.output_dir = output_dir
        
    def load_stock_data(self):
        """Load and preprocess stock price data"""
        print(f"Loading stock data from {self.stock_file}...")
        
        # Load stock data
        df = pd.read_csv(self.stock_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        # Calculate features
        df['Returns'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Change'] = (df['Close'] - df['Open']) / df['Open']
        
        # Log transform volume for better distribution
        df['Log_Volume'] = np.log1p(df['Volume'])
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Volatility
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        print(f"Loaded {len(df)} days of stock data from {df['Date'].min()} to {df['Date'].max()}")
        return df
    
    def load_gdelt_data(self):
        """Load and preprocess GDELT data"""
        print(f"Loading GDELT data from {self.gdelt_file}...")
        
        df = pd.read_csv(self.gdelt_file)
        
        # Filter for specific ticker
        df = df[df['Ticker'] == self.ticker].copy()
        
        if len(df) == 0:
            print(f"WARNING: No GDELT data found for ticker {self.ticker}")
            return pd.DataFrame()
        
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} days of GDELT data from {df['Date'].min()} to {df['Date'].max()}")
        return df
    
    def create_rolling_gdelt_features(self, gdelt_df, windows=[3, 7, 14, 30]):
        """
        Create rolling window features from GDELT data
        
        Args:
            gdelt_df: DataFrame with GDELT data
            windows: List of window sizes in days
        """
        print("Creating rolling GDELT features...")
        
        df = gdelt_df.copy()
        
        for window in windows:
            # Article count features
            df[f'ArticleCount_MA_{window}'] = df['ArticleCount'].rolling(window=window).mean()
            df[f'ArticleCount_Sum_{window}'] = df['ArticleCount'].rolling(window=window).sum()
            df[f'ArticleCount_Max_{window}'] = df['ArticleCount'].rolling(window=window).max()
            df[f'ArticleCount_Std_{window}'] = df['ArticleCount'].rolling(window=window).std()
            
            # Tone features
            df[f'Tone_MA_{window}'] = df['Tone'].rolling(window=window).mean()
            df[f'Tone_Min_{window}'] = df['Tone'].rolling(window=window).min()
            df[f'Tone_Max_{window}'] = df['Tone'].rolling(window=window).max()
            df[f'Tone_Std_{window}'] = df['Tone'].rolling(window=window).std()
            
            # Polarity features
            df[f'Polarity_MA_{window}'] = df['Polarity'].rolling(window=window).mean()
            
            # Word count features
            df[f'WordCount_MA_{window}'] = df['WordCount'].rolling(window=window).mean()
            df[f'WordCount_Sum_{window}'] = df['WordCount'].rolling(window=window).sum()
        
        # Binary indicators
        df['High_Coverage'] = (df['ArticleCount'] > df['ArticleCount'].quantile(0.75)).astype(int)
        df['Negative_Tone'] = (df['Tone'] < 0).astype(int)
        df['Very_Negative_Tone'] = (df['Tone'] < -2).astype(int)
        df['High_Polarity'] = (df['Polarity'] > df['Polarity'].quantile(0.75)).astype(int)
        
        # Days since features
        df['Days_Since_Coverage'] = 0
        last_coverage_idx = -999
        for idx in range(len(df)):
            if df.loc[idx, 'ArticleCount'] > 0:
                last_coverage_idx = idx
            df.loc[idx, 'Days_Since_Coverage'] = idx - last_coverage_idx if last_coverage_idx >= 0 else 999
        
        return df
    
    def merge_data(self, stock_df, gdelt_df):
        """
        Merge stock and GDELT data
        
        Args:
            stock_df: DataFrame with stock data
            gdelt_df: DataFrame with GDELT data
        """
        print("Merging stock and GDELT data...")
        
        # Merge on date (left join to keep all stock dates)
        merged = pd.merge(stock_df, gdelt_df, on='Date', how='left')
        
        # Fill missing GDELT values with 0 (no coverage on that day)
        gdelt_cols = [col for col in merged.columns if col not in stock_df.columns or col == 'Date']
        for col in gdelt_cols:
            if col != 'Date' and col != 'Ticker':
                merged[col] = merged[col].fillna(0)
        
        # Forward fill some rolling features (carry forward last known value)
        rolling_cols = [col for col in merged.columns if '_MA_' in col or '_Sum_' in col or '_Std_' in col]
        for col in rolling_cols:
            merged[col] = merged[col].fillna(method='ffill').fillna(0)
        
        print(f"Merged dataset has {len(merged)} rows")
        return merged
    
    def create_lagged_features(self, df, lag_days=[1, 2, 3, 5, 7]):
        """
        Create lagged features for prediction
        
        Args:
            df: Merged DataFrame
            lag_days: List of lag periods
        """
        print("Creating lagged features...")
        
        result = df.copy()
        
        # Lag GDELT features
        gdelt_features = ['ArticleCount', 'Tone', 'Polarity', 'WordCount',
                         'High_Coverage', 'Negative_Tone', 'Very_Negative_Tone']
        
        for feature in gdelt_features:
            if feature in result.columns:
                for lag in lag_days:
                    result[f'{feature}_Lag_{lag}'] = result[feature].shift(lag)
        
        # Lag stock features
        stock_features = ['Returns', 'Volume_Change', 'Volatility_5', 'Price_Range']
        for feature in stock_features:
            if feature in result.columns:
                for lag in lag_days:
                    result[f'{feature}_Lag_{lag}'] = result[feature].shift(lag)
        
        return result
    
    def identify_event_windows(self, df, volume_threshold_percentile=0, 
                               return_threshold_percentile=0):
        """
        Identify significant events (volume spikes, large price drops)
        
        Args:
            df: Merged DataFrame
            volume_threshold_percentile: Percentile for volume spike detection
            return_threshold_percentile: Percentile for price drop detection
        """
        print("Identifying event windows...")
        
        result = df.copy()
        
        # Calculate thresholds
        volume_threshold = result['Volume'].quantile(volume_threshold_percentile / 100)
        return_threshold = result['Returns'].quantile(return_threshold_percentile / 100)
        
        # Mark events
        result['Volume_Spike'] = (result['Volume'] > volume_threshold).astype(int)
        result['Price_Drop'] = (result['Returns'] < return_threshold).astype(int)
        result['Major_Event'] = ((result['Volume_Spike'] == 1) & 
                                 (result['Price_Drop'] == 1)).astype(int)
        
        # Create event windows (30 days before, 30 days after)
        event_dates = result[result['Major_Event'] == 1]['Date'].values
        
        for event_date in event_dates:
            event_date = pd.Timestamp(event_date)
            pre_window = (result['Date'] >= event_date - timedelta(days=30)) & \
                        (result['Date'] < event_date)
            post_window = (result['Date'] > event_date) & \
                         (result['Date'] <= event_date + timedelta(days=30))
            
            result.loc[pre_window, 'Pre_Event_Window'] = 1
            result.loc[result['Date'] == event_date, 'Event_Day'] = 1
            result.loc[post_window, 'Post_Event_Window'] = 1
        
        # Fill NaN with 0
        for col in ['Pre_Event_Window', 'Event_Day', 'Post_Event_Window']:
            if col not in result.columns:
                result[col] = 0
            else:
                result[col] = result[col].fillna(0).astype(int)
        
        print(f"Found {result['Major_Event'].sum()} major events")
        print(f"Volume spike threshold: {volume_threshold:.0f}")
        print(f"Return drop threshold: {return_threshold:.4f}")
        
        return result
    
    def save_processed_data(self, df, filename_suffix='processed'):
        """Save processed data"""
        output_file = os.path.join(self.output_dir, 
                                   f'{self.ticker}_{filename_suffix}.csv')
        
        # Drop rows with too many NaN values (from initial rolling windows)
        df_clean = df.dropna(subset=['Returns', 'Volume_Change'])
        
        print(f"\nSaving processed data to {output_file}")
        print(f"Final dataset shape: {df_clean.shape}")
        print(f"Columns: {len(df_clean.columns)}")
        print(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
        
        df_clean.to_csv(output_file, index=False)
        
        # Save summary statistics
        summary_file = os.path.join(self.output_dir, 
                                    f'{self.ticker}_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Stock-GDELT Integration Summary for {self.ticker}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total observations: {len(df_clean)}\n")
            f.write(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}\n")
            f.write(f"Features: {len(df_clean.columns)}\n\n")
            
            f.write("Stock Statistics:\n")
            f.write(f"  Avg Daily Volume: {df_clean['Volume'].mean():.0f}\n")
            f.write(f"  Avg Daily Return: {df_clean['Returns'].mean():.4%}\n")
            f.write(f"  Return Volatility: {df_clean['Returns'].std():.4%}\n\n")
            
            if 'ArticleCount' in df_clean.columns:
                f.write("GDELT Statistics:\n")
                f.write(f"  Days with Coverage: {(df_clean['ArticleCount'] > 0).sum()}\n")
                f.write(f"  Avg Articles/Day: {df_clean['ArticleCount'].mean():.2f}\n")
                f.write(f"  Avg Tone: {df_clean['Tone'].mean():.2f}\n")
                f.write(f"  Avg Polarity: {df_clean['Polarity'].mean():.2f}\n\n")
            
            f.write("Event Statistics:\n")
            f.write(f"  Volume Spikes: {df_clean['Volume_Spike'].sum()}\n")
            f.write(f"  Price Drops: {df_clean['Price_Drop'].sum()}\n")
            f.write(f"  Major Events: {df_clean['Major_Event'].sum()}\n")
        
        return output_file
    
    def process(self):
        """
        Run full integration pipeline
        """
        print("\n" + "=" * 70)
        print(f"GDELT-Stock Data Integration for {self.ticker}")
        print("=" * 70 + "\n")
        
        # Load data
        stock_df = self.load_stock_data()
        gdelt_df = self.load_gdelt_data()
        
        # Process GDELT features
        if len(gdelt_df) > 0:
            gdelt_df = self.create_rolling_gdelt_features(gdelt_df)
        
        # Merge datasets
        merged_df = self.merge_data(stock_df, gdelt_df)
        
        # Create lagged features
        merged_df = self.create_lagged_features(merged_df)
        
        # Identify event windows
        merged_df = self.identify_event_windows(merged_df)
        
        # Save results
        output_file = self.save_processed_data(merged_df)
        
        print("\n" + "=" * 70)
        print("Integration Complete!")
        print("=" * 70 + "\n")
        
        return merged_df, output_file


def main():
    """Example usage"""
    
    # Configuration
    ticker = 'GOOGL'
    stock_file = 'trades/data/input/company_trades/GOOG.csv'  # Your stock data file
    gdelt_file = 'media/media_data/output/gkg_company_timeseries.csv'  # Your GDELT file
    
    # Create integrator
    integrator = GDELTStockIntegrator(
        stock_file=stock_file,
        gdelt_file=gdelt_file,
        ticker=ticker
    )
    
    # Process data
    merged_df, output_file = integrator.process()
    
    # Display sample
    print("\nSample of merged data:")
    print(merged_df[['Date', 'Close', 'Volume', 'Returns', 
                     'ArticleCount', 'Tone', 'Major_Event']].head(20))
    
    # Display feature correlation with volume
   # if 'ArticleCount' in merged_df.columns:
    #    print("\nTop features correlated with Volume:")
    #    correlations = merged_df.corr()['Volume'].sort_values(ascending=False)
    #    print(correlations.head(15))


if __name__ == "__main__":
    main()