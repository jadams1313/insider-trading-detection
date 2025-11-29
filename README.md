# Insider Trading Detection Using Deep Learning and Media Data

A data science research project that combines LSTM neural networks, GDELT media data, and stock trading volumes to detect potential insider trading patterns. Developed for DS340W course.

## Project Overview

This research pipeline detects anomalous trading patterns by:
1. **Training LSTM models** on historical stock volume data (baseline vs. media-enhanced)
2. **Integrating GDELT media coverage** as additional features to improve predictions
3. **Applying DTW (Dynamic Time Warping)** to identify suspicious trading patterns
4. **Labeling companies** using litigation data as a proxy for past insider trading activity

The project builds on three foundational papers focusing on FinBERT litigation classification, network analysis of trader relationships, and k-means clustering approaches, with novel modifications that introduce comparative analysis and critical evaluation.

---

## Getting Started

### Prerequisites

**Python 3.9+** with the following packages:

```bash
# Core dependencies
numpy
pandas
matplotlib
scikit-learn

# Deep learning
tensorflow  # or keras
keras
transformers
transformers_interpret

# Data processing
beautifulsoup4
requests
zipfile
datetime
random
yfinance

# Analysis
fastdtw  # for DTW calculations
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/jadams1313/insider-trading-detection
cd insider-trading-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up directory structure**
```
insider-trading-detection/
├── litigation/           # Litigation data crawler
├── media/               # GDELT media processing
│   ├── media_data/
│   │   ├── input/raw/   # Raw GDELT GKG files (.zip)
│   │   └── output/      # Processed media data
├── trades/              # Stock data and LSTM models
│   ├── data/
│   │   ├── input/
│   │   │   ├── company_trades/     # Stock price CSVs
│   │   │   └── media_integrated/   # Merged stock + media data
│   │   └── output/      # Model predictions
├── detection/           # DTW pattern detection
└── README.md
```

---

## 📊 Pipeline Workflow

### Step 0: Litigation classifier

1. Labes labels litigation brought against companies as insider or non-insider. 

**Output:** `litigation_data/combined_litigation.csv`

---

### Step 1: Download Stock Data

Place stock price CSV files in `trades/data/input/company_trades/`

**Required format:**
```
Date,Open,High,Low,Close,Adj Close,Volume
2015-01-01,50.23,51.45,49.80,51.20,51.20,1250000
...
```

**Data sources:** Yahoo Finance

---

### Step 2: Process GDELT Media Data

Download GDELT Global Knowledge Graph (GKG) files and process them:
1. The following process is shown in media.ipynb

**Key steps in notebook:**
1. Load raw GKG `.zip` files from `media_data/input/raw/`
2. Filter for insider trading relevance (legal, financial, corporate themes)
3. Extract company mentions and sentiment metrics
4. Create time series: `ArticleCount`, `Tone`, `Polarity` per company/date

**Output:** `media_data/output/gkg_company_timeseries.csv`

---

### Step 3: Integrate Media + Stock Data

Merge GDELT coverage with stock prices

1. You will find any necessary integration steps in MediaIntegration.ipynb


### Step 4: Train LSTM Models

1. Train both baseline (volume-only) and enhanced (volume + media) models:
2. Any necessary training information can be found in trades.ipynb
3. Outputs for performance are saved as PNGs locally, replicating similar statistical measurements and visuals as the parent paper. 

**Configuration:**
```python
integrated_file = 'data/input/media_integrated/AAPL_processed.csv'
company = 'AAPL'
seq_len = 50      # Look-back window
epochs = 20       # Training epochs
```

**Training process:**
```python
from trades.lstm import EnhancedLSTMPipeline

pipeline = EnhancedLSTMPipeline(
    integrated_file=integrated_file,
    seq_len=seq_len,
    company=company
)

# Train baseline model (volume patterns only)
model_baseline, history_baseline = pipeline.train_and_predict(
    use_gdelt=False, 
    epochs=epochs
)

# Train enhanced model (volume + GDELT features)
model_enhanced, history_enhanced = pipeline.train_and_predict(
    use_gdelt=True, 
    epochs=epochs
)
```

**Outputs:** Six prediction files per company:
- `data/output/AAPL_full_window_pred_baseline.csv`
- `data/output/AAPL_full_point_pred_baseline.csv`
- `data/output/AAPL_full_sequence_pred_baseline.csv`
- `data/output/AAPL_full_window_pred_enhanced.csv`
- `data/output/AAPL_full_point_pred_enhanced.csv`
- `data/output/AAPL_full_sequence_pred_enhanced.csv`

---

### Step 5: Detect Anomalies with DTW
1. You will find any necessary detection steps in detect.ipynb
---

## 📁 Key Files & Modules

### Litigation Module
- **`litigation/insider_data_crawler.py`**: Scrapes SEC, FBI, and US SDNY litigation releases
- **`litigation/litigation_classifier.py`** Progressively evaluates Litgation data to identify those under suspicion for insider trading. 

### Media Module
- **`media/media.ipynb`**: GDELT GKG processing notebook
- **`media/media_data.py`**: Helper functions for parsing and filtering GKG data

### Trades Module
- **`trades/trades.ipynb`**: LSTM training notebook
- **`trades/lstm.py`**: LSTM pipeline for baseline and enhanced models
- **`trades/media_integration.py`**: Merges GDELT and stock data

### Detection Module
- **`detection/detection.ipynb`**: Pattern detection notebook
- **`detection/detection_tools.py`**: DTW pattern matching and analysis tools

---

## 🔍 Methodology

### 1. Baseline Model
- **Input:** Historical trading volume sequences
- **Architecture:** 2-layer LSTM (100 units each) with dropout
- **Prediction:** Next-day volume using three methods (window/point/sequence)

### 2. Enhanced Model
- **Input:** Volume + GDELT media features (article count, sentiment, tone)
- **Architecture:** LSTM 
- **Hypothesis:** Media coverage improves volume prediction accuracy

### 3. Pattern Detection
- **Method:** Dynamic Time Warping (DTW) distance
- **Patterns:** Synthetic insider trading signatures (5-25 day patterns)
  - Gradual volume increases
  - Sharp spikes followed by drops
  - Sustained elevated volume
- **Threshold:** Empirically determined (0.5 baseline, 0.45 enhanced)
- **Output:** Anomalies flagged by method, ticker, day, and pattern type

### 4. Evaluation
- **Metrics:** R², RMSE, MAE, MAPE (prediction quality)
- **Comparison:** Baseline vs. enhanced model performance
- **Detection rate:** Number of anomalies by method and ticker

---

---

## 🎯 Research Questions

1. **Does media coverage improve volume prediction accuracy? Consequently, does it improve detection accuracy?**
   - Compare R² and RMSE between baseline and enhanced models

2. **Which prediction method detects the most anomalies?**
   - Analyze detection counts across window/point/sequence methods

3. **Do enhanced models reduce false positives?**
   - Compare detection rates and DTW distance distributions

---

## 🛠️ Troubleshooting

### Common Issues

**Issue:** LSTM model not improving
- Check data normalization (should normalize each window relative to first value)
- Verify no data leakage between train/test splits
- Increase epochs or adjust learning rate

**Issue:** No anomalies detected
- Lower DTW threshold 
- Check that patterns match data scale (both should be normalized)
- Verify data loading: `dt.load_ticker_data()` returns correct structure

**Issue:** GDELT data missing for ticker
- Confirm ticker appears in `media_data/output/gkg_company_timeseries.csv`
- Check spelling/format (e.g., 'GOOGL' not 'GOOG')
- Ensure GDELT files cover the date range of your stock data

**Issue:** Media integration produces NaN values
- Fill missing GDELT days with zeros: `ArticleCount=0`, `Tone=0`, `Polarity=0`
- Use forward-fill for moving averages
- Check date alignment between stock and GDELT files

---

## 📚 References

1. **FinBERT-based litigation classification** - Text analysis for legal documents


---

---

## 📝 License

This project is for educational and research purposes only.

---

## 🔗 Resources

- **GDELT Project:** https://www.gdeltproject.org/
- **Yahoo Finance:** https://finance.yahoo.com/
- **SEC Litigation Releases:** https://www.sec.gov/litigation/litreleases
- **Keras Documentation:** https://keras.io/
- **DTAIDistance (DTW):** https://dtaidistance.readthedocs.io/

---

**Last Updated:** November 2025
