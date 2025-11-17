import numpy as np
import pandas as pd
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
import warnings
warnings.filterwarnings("ignore")

class EnhancedLSTMPipeline:
    """
    Pipeline to train baseline and GDELT-enhanced models
    Outputs predictions in format compatible with detection.py
    """
    
    def __init__(self, integrated_file, seq_len=50, company='amsc'):
        self.integrated_file = integrated_file
        self.seq_len = seq_len
        self.company = company
        self.output_dir = 'data/output'
    def calculate_metrics(self, predictions, actuals):
        """Calculate error metrics"""
        # Remove any NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]
        
        if len(predictions) == 0:
            return None
        
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # MAPE (avoiding division by zero)
        mask = actuals != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        else:
            mape = np.nan
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'N': len(predictions)
        }
    
        
    def load_and_prepare_data(self, use_gdelt=False):
        """
        Load integrated data and prepare for LSTM
        
        Args:
            use_gdelt: If True, include GDELT features; if False, baseline only
        """
        print(f"\n{'='*70}")
        print(f"Loading data - {'ENHANCED (with GDELT)' if use_gdelt else 'BASELINE (volume only)'}")
        print(f"{'='*70}")
        
        # Load integrated data
        df = pd.read_csv(self.integrated_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Total observations: {len(df)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Define feature sets
        baseline_features = [
            'Volume', 'Returns', 'Volume_Change', 
            'MA_5', 'MA_20', 'Volatility_5', 'Price_Range'
        ]
        
        gdelt_features = [
            'ArticleCount', 'Tone', 'Polarity',
            'ArticleCount_MA_7', 'Tone_MA_7',
            'High_Coverage', 'Negative_Tone', 'Very_Negative_Tone'
        ]
        
        # Select features based on mode
        if use_gdelt:
            feature_cols = baseline_features + [f for f in gdelt_features if f in df.columns]
        else:
            feature_cols = baseline_features
        
        # Ensure all features exist
        feature_cols = [f for f in feature_cols if f in df.columns]
        
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Extract data
        data = df[feature_cols].values
        
        # Remove NaN rows
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        
        print(f"After removing NaN: {data.shape[0]} samples")
        
        # Get volume index (target variable)
        volume_idx = feature_cols.index('Volume')
        
        return data, volume_idx, feature_cols
    
    def normalize_windows(self, window_data):
        """
        Normalize each window relative to first value
        (matches your existing normalization approach)
        """
        normalized_data = []
        for window in window_data:
            if window[0, 0] != 0:  # Avoid division by zero
                normalized_window = (window / window[0, 0]) - 1
            else:
                normalized_window = window
            normalized_data.append(normalized_window)
        return np.array(normalized_data)
    
    def create_sequences(self, data, volume_idx):
        """
        Create sequences for LSTM training
        Target is next day's volume
        """
        sequence_length = self.seq_len + 1
        result = []
        
        for i in range(len(data) - sequence_length):
            result.append(data[i:i + sequence_length])
        
        result = np.array(result)
        
        # Normalize windows
        result = self.normalize_windows(result)
        
        # Split data: 70% train, 20% test, 10% validation
        total_samples = result.shape[0]
        train_end = int(0.7 * total_samples)
        test_end = int(0.9 * total_samples)
        
        train = result[:train_end]
        test = result[train_end:test_end]
        val = result[test_end:]
        
        # Separate features and target (volume)
        X_train = train[:, :-1, :]
        y_train = train[:, -1, volume_idx]
        
        X_test = test[:, :-1, :]
        y_test = test[:, -1, volume_idx]
        
        X_valid = val[:, :-1, :]
        y_valid = val[:, -1, volume_idx]
        
        print(f"\nData split:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Testing: {X_test.shape[0]} samples")
        print(f"  Validation: {X_valid.shape[0]} samples")
        
        return X_train, y_train, X_test, y_test, X_valid, y_valid
    
    def build_lstm_model(self, n_features, lstm_units=100):
        """
        Build LSTM model matching your existing architecture
        """
        model = Sequential()
        
        model.add(LSTM(units=lstm_units, 
                       input_shape=(self.seq_len, n_features),
                       return_sequences=True))
        
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(0.2))
        
        model.add(Dense(units=1))
        model.add(Activation("linear"))
        
        model.compile(loss="mse", optimizer="rmsprop")
        
        return model
    
    def predict_point_by_point(self, model, data):
        """Point-by-point prediction (day ahead forecasting)"""
        predicted = model.predict(data, verbose=0)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
    
    def predict_sequence_full(self, model, data):
        """Historical-based forecasting (entire history)"""
        curr_frame = data[0]
        predicted = []
        
        for i in range(len(data)):
            pred = model.predict(curr_frame[np.newaxis, :, :], verbose=0)[0, 0]
            predicted.append(pred)
            curr_frame = np.roll(curr_frame, -1, axis=0)
            curr_frame[-1] = pred
        
        return np.array(predicted)
    
    def predict_sequences_multiple(self, model, data, prediction_len=50):
        """Window-based forecasting"""
        predictions = []
        curr_frame = data[0]

        for i in range(len(data)):
            pred = model.predict(curr_frame[np.newaxis, :, :], verbose=0)[0, 0]
            predictions.append(pred)

            curr_frame = np.roll(curr_frame, -1, axis=0)
            curr_frame[-1] = pred

        return np.array(predictions)
    
    def save_predictions(self, predictions, y_test, method_name, model_type):
        """
        Save predictions in format compatible with detection.py
        
        Args:
            predictions: Model predictions
            y_test: Actual values
            method_name: 'window', 'point', or 'sequence'
            model_type: 'baseline' or 'enhanced'
        """
        # Flatten window-based predictions
        if method_name == 'window':
            if isinstance(predictions[0], (list, np.ndarray)):
                predictions = np.array([item for sublist in predictions for item in sublist])
            else:
                # Already flat (1D), do nothing
                predictions = np.array(predictions)
        
        # Save predictions
        pred_file = f"{self.output_dir}/{self.company}_full_{method_name}_pred_{model_type}.csv"
        pd.DataFrame(predictions).to_csv(pred_file, index=False, header=False)
        
        # Save actuals (same for both models)
        if model_type == 'baseline':
            actual_file = f"{self.output_dir}/{self.company}_full_{method_name}_act.csv"
            pd.DataFrame(y_test).to_csv(actual_file, index=False, header=False)
        
        print(f"  Saved: {pred_file}")
    
    def train_and_predict(self, use_gdelt=False, epochs=20):
        """
        Main training and prediction pipeline
        
        Args:
            use_gdelt: If True, include GDELT features
            epochs: Number of training epochs
        """
        model_type = 'enhanced' if use_gdelt else 'baseline'
        start_time = time.time()
        
        # Load and prepare data
        data, volume_idx, feature_cols = self.load_and_prepare_data(use_gdelt)
        
        # Create sequences
        X_train, y_train, X_test, y_test, X_valid, y_valid = self.create_sequences(
            data, volume_idx
        )
        
        # Build model
        print(f"\nBuilding LSTM model...")
        model = self.build_lstm_model(n_features=X_train.shape[2])
        
        # Train model
        print(f"\nTraining for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            batch_size=512,
            epochs=epochs,
            validation_data=(X_valid, y_valid),
            verbose=1
        )
        
        print(f"\nGenerating predictions for all three methods...")
        
        # Method 1: window-based 
        print("  1. Window-based forecasting...")
        predictions_window = self.predict_sequences_multiple(model, X_test, prediction_len=50)
        self.save_predictions(predictions_window, y_test, 'window', model_type)
        
        # Method 2:(day ahead)
        print("  2. Day-ahead forecasting...")
        predictions_point = self.predict_point_by_point(model, X_test)
        self.save_predictions(predictions_point, y_test, 'point', model_type)
        
        # Method 3: (historical)
        print("  3. Historical-based forecasting...")
        predictions_sequence = self.predict_sequence_full(model, X_test)
        self.save_predictions(predictions_sequence, y_test, 'sequence', model_type)
        
        elapsed = time.time() - start_time
        print(f"\n{model_type.upper()} model complete! Time: {elapsed:.1f}s")
        
        return model, history
    
    def run_full_pipeline(self, epochs=20):
        """
        Run both baseline and enhanced models
        Generate all predictions needed for detection.py
        """
        print("\n" + "="*70)
        print("ENHANCED LSTM PIPELINE - BASELINE vs GDELT-ENHANCED")
        print("="*70)
        
        # Train baseline model
        print("\n>>> TRAINING BASELINE MODEL (Volume patterns only)")
        model_baseline, history_baseline = self.train_and_predict(
            use_gdelt=False, 
            epochs=epochs
        )
        
        # Train enhanced model
        print("\n>>> TRAINING ENHANCED MODEL (Volume + GDELT features)")
        model_enhanced, history_enhanced = self.train_and_predict(
            use_gdelt=True, 
            epochs=epochs
        )
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print("\nGenerated files for detection.py:")
        print("  Baseline predictions:")
        print("    - data/amsc_full_window_pred_baseline.csv")
        print("    - data/amsc_full_point_pred_baseline.csv")
        print("    - data/amsc_full_sequence_pred_baseline.csv")
        print("\n  Enhanced predictions:")
        print("    - data/amsc_full_window_pred_enhanced.csv")
        print("    - data/amsc_full_point_pred_enhanced.csv")
        print("    - data/amsc_full_sequence_pred_enhanced.csv")
        print("\n  Actuals (shared):")
        print("    - data/amsc_full_window_act.csv")
        print("    - data/amsc_full_point_act.csv")
        print("    - data/amsc_full_sequence_act.csv")
        print("\nNext step: Run detection.py on both prediction sets")
        
        return model_baseline, model_enhanced


def main():
    """Main execution"""
    
    # Configuration
    integrated_file = 'data/AMSC_processed.csv'
    company = 'amsc'
    seq_len = 50
    epochs = 20
    
    # Check if integrated data exists
    if not os.path.exists(integrated_file):
        print(f"ERROR: Integrated data not found: {integrated_file}")
        print("Please run gdelt_integration.py first!")
        return
    
    # Run pipeline
    pipeline = EnhancedLSTMPipeline(
        integrated_file=integrated_file,
        seq_len=seq_len,
        company=company
    )
    
    model_baseline, model_enhanced = pipeline.run_full_pipeline(epochs=epochs)
    


if __name__ == "__main__":
    main()