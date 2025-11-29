import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import os


# 1. DATA LOADING
def load_time_series(filepath):
    """Load single time series from CSV"""
    return pd.read_csv(filepath, header=None).iloc[:, 0].values


def load_ticker_data(ticker, model_type, data_dir='data'):
    """
    Load all prediction variants for a ticker
    
    Returns: dict with 'actual', 'point', 'sequence', 'window' arrays
    """
    base = f"{data_dir}/{ticker}_full"
    return {
        'actual': load_time_series(f"{base}_point_act.csv"),
        'point': load_time_series(f"{base}_point_pred_{model_type}.csv"),
        'sequence': load_time_series(f"{base}_sequence_pred_{model_type}.csv"),
        'window': load_time_series(f"{base}_window_pred_{model_type}.csv")
    }


# 2. PATTERN GENERATION
def generate_insider_patterns(lengths=[10, 15, 20, 25, 30, 35, 40, 45, 50]):
    """
    Generate synthetic insider trading patterns
    
    Pattern simulates:
    - Gradual volume increase (accumulation phase)
    - Sharp spike (announcement/event)
    - Decay (post-event)
    
    Returns: dict {pattern_id: normalized_array}
    """
    patterns = {}
    
    for i, length in enumerate(lengths, 1):
        t = np.linspace(0, 1, length)
        
        # Composite pattern
        pattern = (
            0.5 * t +                          # Linear increase
            3.0 * np.exp(-10 * (t - 0.7)**2)   # Gaussian spike at 70%
        )
        
        # Normalize
        patterns[i] = (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-8)
    
    return patterns


# 3. DTW COMPUTATION
def normalize_signal(signal):
    """Z-score normalization"""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def compute_dtw(signal1, signal2):
    """
    Compute DTW distance between two normalized signals
    
    Returns: (distance, alignment_path)
    """
    s1 = normalize_signal(signal1)
    s2 = normalize_signal(signal2)
    return fastdtw(s1, s2, dist=lambda x, y: abs(x - y))


# 4. THRESHOLD DETERMINATION
def sample_dtw_distances(data, patterns, window_size=50, overlap=10, 
                        sample_rate=0.2):
    """
    Sample DTW distances to understand distribution
    
    Args:
        sample_rate: Fraction of possible comparisons to sample
        
    Returns: list of DTW distances
    """
    distances = []
    num_windows = int(np.ceil(len(data) / window_size))
    
    for window_idx in range(num_windows):
        start = window_idx * window_size
        window_data = data[start:start + window_size]
        current_overlap = 0 if window_idx == num_windows - 1 else overlap
        
        for pattern in patterns.values():
            max_day = len(window_data) - len(pattern) + current_overlap
            
            if max_day <= 0:
                continue
            
            # Randomly sample positions
            n_samples = max(1, int(max_day * sample_rate))
            for day in np.random.choice(max_day, size=min(n_samples, max_day), 
                                       replace=False):
                signal = window_data[day:day + len(pattern)]
                if len(signal) == len(pattern):
                    dist, _ = compute_dtw(signal, pattern)
                    distances.append(dist)
    
    return distances


def get_threshold_stats(distances):
    """Calculate distribution statistics"""
    return {
        'min': np.min(distances),
        'p10': np.percentile(distances, 10),
        'p25': np.percentile(distances, 25),
        'median': np.median(distances),
        'p75': np.percentile(distances, 75),
        'max': np.max(distances)
    }


def suggest_threshold(distances, percentile=10, verbose=True):
    """Suggest threshold based on percentile"""
    threshold = np.percentile(distances, percentile)
    
    if verbose:
        stats = get_threshold_stats(distances)
        print("\nDTW Distance Distribution:")
        for key, val in stats.items():
            print(f"  {key:8s}: {val:6.3f}")
        print(f"\nSuggested threshold ({percentile}th %ile): {threshold:.3f}")
    
    return threshold


# 5. ANOMALY DETECTION
def detect_anomalies(data, patterns, threshold, window_size=50, overlap=10):
    """
    Scan time series for pattern matches below threshold
    
    Returns: list of anomaly dicts
    """
    anomalies = []
    num_windows = int(np.ceil(len(data) / window_size))
    
    for window_idx in range(num_windows):
        start = window_idx * window_size
        window_data = data[start:start + window_size]
        current_overlap = 0 if window_idx == num_windows - 1 else overlap
        
        for pattern_id, pattern in patterns.items():
            pattern_len = len(pattern)
            max_day = len(window_data) - pattern_len + current_overlap
            
            for day in range(max_day):
                signal = window_data[day:day + pattern_len]
                
                if len(signal) < pattern_len:
                    continue
                
                distance, path = compute_dtw(signal, pattern)
                
                if distance <= threshold:
                    anomalies.append({
                        'window': window_idx + 1,
                        'pattern_id': pattern_id,
                        'pattern_length': pattern_len,
                        'day_in_window': day,
                        'actual_day': window_size + (window_idx * window_size) + day,
                        'dtw_distance': distance,
                        'signal': signal,
                        'pattern': pattern,
                        'path': path
                    })
    
    return anomalies


def run_detection(data_dict, patterns, threshold, window_size=50, overlap=10):
    """Run detection across all prediction methods"""
    methods = {
        'Point-by-Point': data_dict['point'],
        'Sequence': data_dict['sequence'],
        'Window': data_dict['window']
    }
    
    results = {}
    for name, data in methods.items():
        anomalies = detect_anomalies(data, patterns, threshold, window_size, overlap)
        results[name] = anomalies
        print(f"{name:20s}: {len(anomalies):4d} anomalies")
    
    return results


# ANALYSIS & REPORTING
def summarize_results(results):
    """Create summary DataFrame"""
    rows = []
    
    for method, anomalies in results.items():
        if not anomalies:
            continue
        
        df = pd.DataFrame(anomalies)
        rows.append({
            'Method': method,
            'Count': len(anomalies),
            'DTW Min': df['dtw_distance'].min(),
            'DTW Mean': df['dtw_distance'].mean(),
            'DTW Max': df['dtw_distance'].max(),
            'Patterns': df['pattern_id'].nunique()
        })
    
    return pd.DataFrame(rows)


def pattern_breakdown(results):
    """Count detections per pattern"""
    all_data = []
    
    for method, anomalies in results.items():
        for a in anomalies:
            all_data.append({
                'method': method,
                'pattern_id': a['pattern_id'],
                'dtw': a['dtw_distance']
            })
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    
    # Aggregate by pattern
    summary = df.groupby('pattern_id').agg({
        'method': 'count',
        'dtw': 'mean'
    }).rename(columns={'method': 'total', 'dtw': 'avg_dtw'})
    
    # Method breakdown
    breakdown = df.groupby(['pattern_id', 'method']).size().unstack(fill_value=0)
    
    return pd.concat([summary, breakdown], axis=1).sort_values('total', ascending=False)


# 7. VISUALIZATION
def plot_distance_distribution(all_results, threshold=None, bins=50, method='baseline'):
    """Histogram of DTW distances across all tickers for a given method"""
    distances = []
    
    # Collect all distances from all tickers
    for ticker, results in all_results.items():
        for method_key, anomalies in results.items():
            distances.extend([a['dtw_distance'] for a in anomalies])
    
    if not distances:
        print(f"No distances to plot for {method} method")
        return
    
    plt.figure(figsize=(10, 5))
    plt.hist(distances, bins=bins, alpha=0.7, edgecolor='black', color='steelblue')
    
    if threshold:
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold:.2f}')
        plt.legend()
    
    plt.xlabel('DTW Distance')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of DTW Distances - {method.upper()} Method')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_method_comparison(all_results, ticker=None):
    """Bar chart comparing detection methods
    
    If ticker is provided, shows comparison for that ticker only.
    Otherwise, shows aggregated comparison across all tickers.
    """
    if ticker:
        # Single ticker comparison
        if ticker not in all_results:
            print(f"No results for ticker {ticker}")
            return
        
        results = all_results[ticker]
        methods = list(results.keys())
        counts = [len(results[m]) for m in methods]
        title = f'Detection Count by Method - {ticker}'
    else:
        # Aggregate across all tickers
        method_counts = {}
        for ticker_results in all_results.values():
            for method, anomalies in ticker_results.items():
                method_counts[method] = method_counts.get(method, 0) + len(anomalies)
        
        methods = list(method_counts.keys())
        counts = [method_counts[m] for m in methods]
        title = 'Detection Count by Method - All Tickers'
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, counts, alpha=0.7, edgecolor='black', color='steelblue')
    
    plt.xlabel('Prediction Method')
    plt.ylabel('Anomalies Detected')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2, height + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_ticker_comparison(all_results, method_name=None):
    """Bar chart comparing detection across tickers
    
    If method_name is provided, shows that method only.
    Otherwise, shows total across all methods.
    """
    tickers = list(all_results.keys())
    
    if method_name:
        counts = [len(all_results[t].get(method_name, [])) for t in tickers]
        title = f'Detection Count by Ticker - {method_name} Method'
    else:
        counts = [sum(len(anomalies) for anomalies in all_results[t].values()) 
                  for t in tickers]
        title = 'Total Detection Count by Ticker - All Methods'
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(tickers, counts, alpha=0.7, edgecolor='black', color='coral')
    
    plt.xlabel('Ticker')
    plt.ylabel('Anomalies Detected')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2, height + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_top_anomalies(all_results, ticker, method_name, top_n=5, window_size=30):
    """Plot top N anomalies for a specific ticker and method
    
    Args:
        all_results: Dictionary of results by ticker and method
        ticker: Stock ticker symbol
        method_name: Detection method name
        top_n: Number of top anomalies to plot
        window_size: Window size used in detection (for day calculation)
    """
    if ticker not in all_results:
        print(f"No results for ticker {ticker}")
        return
    
    if method_name not in all_results[ticker] or not all_results[ticker][method_name]:
        print(f"No anomalies for {ticker} - {method_name}")
        return
    
    anomalies = sorted(all_results[ticker][method_name], 
                      key=lambda x: x['dtw_distance'])[:top_n]
    
    fig, axes = plt.subplots(top_n, 1, figsize=(12, 2.5*top_n))
    if top_n == 1:
        axes = [axes]
    
    for idx, anom in enumerate(anomalies):
        ax = axes[idx]
        
        # Calculate exact day: day = window_size + (w * window_size) + d
        w = anom.get('window_idx', 0)  # Window index
        d = anom.get('day_in_window', 0)  # Day within window
        exact_day = window_size + (w * window_size) + d
        
        ax.plot(anom['signal'], 'b-', linewidth=2, alpha=0.7, label='Predicted')
        ax.plot(anom['pattern'], 'r-', linewidth=2, alpha=0.7, label='Pattern')
        
        # Enhanced title with exact day calculation
        title_parts = [
            f"#{idx+1}: Pattern {anom['pattern_id']}",
            f"Exact Day: {exact_day}",
            f"(Window {w}, Day {d})",
            f"DTW={anom['dtw_distance']:.3f}"
        ]
        ax.set_title(" | ".join(title_parts))
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (Days in Pattern)')
        ax.set_ylabel('Norm. Volume')
    
    plt.suptitle(f'{ticker} - {method_name} - Top {top_n} Anomalies (Lowest DTW)', 
                fontsize=13, y=1.001)
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print(f"\n{ticker} - {method_name} - Top {top_n} Anomalies:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Pattern':<10} {'Window':<8} {'Day':<6} {'Exact Day':<12} {'DTW':<10}")
    print("-" * 80)
    for idx, anom in enumerate(anomalies):
        w = anom.get('window_idx', 0)
        d = anom.get('day_in_window', 0)
        exact_day = window_size + (w * window_size) + d
        print(f"{idx+1:<6} {anom['pattern_id']:<10} {w:<8} {d:<6} {exact_day:<12} {anom['dtw_distance']:<10.3f}")