import yfinance as yf
import pandas as pd
import time

# Configuration
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
TICKERS = ['AMSC', 'BP', 'EVR', 'GOOGL', 'GTX', 'HLF', 'MDRX', 'ORCL', 'SPPI', 'WFC']

def download_ticker(ticker):
    """Download daily data for a single ticker and save to CSV"""
    try:
        print(f"Downloading {ticker}...")
        
        # Download the data
        stock = yf.Ticker(ticker)
        if ticker == 'SPPI': 
            df = stock.history(
                start=START_DATE,
                end="2022-12-31",
                interval='1d',
                actions=False
            )
        else: 
            df = stock.history(
                start=START_DATE,
                end=END_DATE,
                interval='1d'
            )
        
        if df.empty:
            print(f"  ⚠ No data found for {ticker}")
            return False
        
        # Save to CSV
        filename = f"{ticker}_2015_2025.csv"
        df.to_csv(filename)
        
        print(f"  ✓ Saved {len(df)} days to {filename}")
        time.sleep(0.5)  # Be nice to the API
        return True
        
    except Exception as e:
        print(f"  ✗ Error with {ticker}: {str(e)}")
        return False


if __name__ == "__main__":
    print(f"Downloading {len(TICKERS)} tickers from {START_DATE} to {END_DATE}\n")
    
    successful = 0
    for i, ticker in enumerate(TICKERS, 1):
        print(f"[{i}/{len(TICKERS)}]", end=" ")
        if download_ticker(ticker):
            successful += 1
    
    print(f"\n{'='*50}")
    print(f"Complete! Downloaded {successful}/{len(TICKERS)} tickers")
    print(f"CSV files saved to current directory")