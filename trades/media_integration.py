import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MediaStockFeatureEngineer:
    def __init__(self, mentions_csv, stock_data_dir, tickers):
        """
        Args:
            mentions_csv: Path to the media mentions CSV
            stock_data_dir: Directory containing stock CSV files
            tickers: List of ticker symbols
        """
        self.mentions_csv = mentions_csv
        self.stock_data_dir = Path(stock_data_dir)
        self.tickers = tickers
        
    def load_mentions_data(self):
        """Load and prepare media mentions data"""
        logger.info(f"Loading mentions data from {self.mentions_csv}")
        
        df = pd.read_csv(self.mentions_csv)
        # Convert to datetime and normalize to date only (no time component)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.normalize()
        
        logger.info(f"Loaded {len(df)} mention records")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Tickers: {df['Ticker'].unique().tolist()}")
        
        return df
    
    def load_stock_data(self):
        """Load all stock CSV files and combine"""
        logger.info(f"Loading stock data from {self.stock_data_dir}")
        
        all_stock_data = []
        
        for ticker in self.tickers:
            # Try common filename patterns
            possible_files = [
                self.stock_data_dir / f"{ticker}_2015_2025.csv",
                self.stock_data_dir / f"{ticker}.csv",
            ]
            
            stock_file = None
            for f in possible_files:
                if f.exists():
                    stock_file = f
                    break
            
            if stock_file is None:
                logger.warning(f"Could not find stock data for {ticker}")
                continue
            
            logger.info(f"Loading {ticker} from {stock_file.name}")
            df = pd.read_csv(stock_file)
            
            # Handle date column (might be index or column)
            if 'Date' not in df.columns:
                df = df.reset_index()
            
            # Convert to datetime and strip timezone, keep only date part
            df['Date'] = pd.to_datetime(df['Date'],utc=True).dt.tz_localize(None)
            df['Ticker'] = ticker
            
            # Keep only relevant columns
            columns_to_keep = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            all_stock_data.append(df)
        
        combined = pd.concat(all_stock_data, ignore_index=True)
        logger.info(f"Combined stock data: {len(combined)} records across {len(all_stock_data)} tickers")
        
        return combined
    
    def merge_data(self, mentions_df, stock_df, ticker='GOOGL'):
        """Merge mentions and stock data"""
        logger.info("Merging mentions and stock data...")
        
        # Filter mentions data for the specific ticker
        mentions_df = mentions_df[mentions_df['Ticker'] == ticker].copy()
        
        # Ensure Date columns are datetime, timezone-naive, and normalized to midnight
        mentions_df['Date'] = pd.to_datetime(mentions_df['Date'], utc=True).dt.tz_localize(None)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True).dt.tz_localize(None)
        
        logger.info(f"Stock date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
        logger.info(f"Mentions date range: {mentions_df['Date'].min()} to {mentions_df['Date'].max()}")
                
        # Drop Ticker column from mentions before merge (to avoid duplication)
        mentions_df = mentions_df.drop(columns=['Ticker'])
        
        # Left join: keep all stock trading days, add mentions data
        merged = stock_df.merge(
            mentions_df,
            on='Date',
            how='left'
        )
        
        logger.info(f"After mentions merge: {len(merged)} records, {merged['ArticleCount'].notna().sum()} with mentions")
        
        # Fill missing mention data with zeros
        mention_columns = ['ArticleCount', 'Tone', 'Polarity', 'WordCount']
        for col in mention_columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        logger.info(f"Final merged data: {len(merged)} records")
        
        return merged
    
    def create_target_variable(self, df):
        """Create Volume_Next_Day target variable"""
        logger.info("Creating target variable (Volume_Next_Day)...")
        
        df = df.sort_values(['Ticker', 'Date'])
        
        # Shift volume by -1 within each ticker group
        df['Volume'] = df.groupby('Ticker')['Volume'].shift(-1)
        
        logger.info(f"Target variable created. {df['Volume'].isna().sum()} missing values (last day per ticker)")
        
        return df
    
    def create_lag_features(self, df, columns, lags=[1, 7, 30]):
        """Create lagged features"""
        logger.info(f"Creating lag features for {columns}...")
        
        df = df.sort_values(['Ticker', 'Date'])
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                new_col = f"{col}_lag{lag}"
                df[new_col] = df.groupby('Ticker')[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, columns, window=30):
        """Create rolling window features"""
        logger.info(f"Creating {window}-day rolling window features...")
        
        df = df.sort_values(['Ticker', 'Date'])
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Rolling mean
            df[f"{col}_mean_{window}d"] = (
                df.groupby('Ticker')[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # Rolling std
            df[f"{col}_std_{window}d"] = (
                df.groupby('Ticker')[col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
            
            # Spike detection (z-score)
            mean_col = f"{col}_mean_{window}d"
            std_col = f"{col}_std_{window}d"
            df[f"{col}_spike"] = (df[col] - df[mean_col]) / (df[std_col] + 1e-8)  # Avoid division by zero
        
        return df
    
    def create_temporal_features(self, df):
        """Create temporal features"""
        logger.info("Creating temporal features...")
        
        df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['year'] = df['Date'].dt.year
        
        # Is last day of month/quarter
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
        
        return df
    
    def engineer_features(self):
        """Run complete feature engineering pipeline"""
        logger.info("="*60)
        logger.info("Starting feature engineering pipeline")
        logger.info("="*60)
        
        # Load data
        mentions_df = self.load_mentions_data()
        stock_df = self.load_stock_data()
        
        # Merge
        df = self.merge_data(mentions_df, stock_df)
        
        # Create target variable
        df = self.create_target_variable(df)
        
        # Create lag features
        media_cols = ['ArticleCount', 'Tone', 'Polarity', 'WordCount']
        stock_cols = ['Volume', 'Close']
        df = self.create_lag_features(df, media_cols + stock_cols, lags=[1, 7, 30])
        
        # Create rolling features
        df = self.create_rolling_features(df, media_cols + stock_cols, window=30)
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Sort by date and ticker
        df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        
        logger.info("="*60)
        logger.info("Feature engineering complete!")
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info("="*60)
        
        return df
    
    def save_dataset(self, df, output_file):
        """Save engineered dataset to CSV"""
        logger.info(f"Saving dataset to {output_file}")
        df.to_csv(output_file, index=False)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Tickers: {df['Ticker'].unique().tolist()}")
        print(f"Features: {len(df.columns)}")
        print(f"\nMissing values in target (Volume_Next_Day): {df['Volume_Next_Day'].isna().sum()}")
        print(f"\nFirst few rows:")
        print(df.head(10))
        print("\nColumn list:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        print("="*60)


if __name__ == "__main__":
    # Configuration
    MENTIONS_CSV = "media_mentions.csv"  # Your mentions CSV file
    STOCK_DATA_DIR = "."  # Directory with stock CSV files
    TICKERS = ['AMSC', 'BP', 'EVR', 'GOOGL', 'GTXI', 'HLF', 'MDRX', 'ORCL', 'SPPI', 'WFC']
    OUTPUT_FILE = "media_stock_features.csv"
    
    # Run feature engineering
    engineer = MediaStockFeatureEngineer(
        mentions_csv=MENTIONS_CSV,
        stock_data_dir=STOCK_DATA_DIR,
        tickers=TICKERS
    )
    
    df = engineer.engineer_features()
    engineer.save_dataset(df, OUTPUT_FILE)
    
    print(f"\n✓ Feature engineering complete!")
    print(f"✓ Output saved to: {OUTPUT_FILE}")