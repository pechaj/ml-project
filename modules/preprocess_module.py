import numpy as np
import pandas as pd
from neurokit2 import ecg_clean, eda_phasic, eda_clean, ecg_peaks

def preprocessDataset(ecg_signal_full: pd.DataFrame, eda_signal_full: pd.DataFrame, fs: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    
    # Extract ECG signal and remove NaN values
    ecg_signal_full = ecg_signal_full.dropna()
    eda_signal_full = eda_signal_full.dropna()
    
    ecg_signal_filtered, ecg_signal_raw = preprocessSignalECG(ecg_signal_full, fs)
    eda_signal_filtered, eda_signal_raw = preprocessSignalEDA(eda_signal_full, fs)
    
    return ecg_signal_filtered, eda_signal_filtered, ecg_signal_raw, eda_signal_raw
    
def preprocessSignalECG(signal: pd.DataFrame, fs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    
    if "ecg2" not in signal.columns:
            raise ValueError("Column 'ecg2' not found in the dataset.")
        
    ecg_signal_ecg = signal["ecg2"].to_numpy()
    
    if len(ecg_signal_ecg) < 1024:
        print("ECG signal is too short, skipping processing.")
        return None, None
    
    try:
        # Try to process the ECG signal
        ecg_signal_processed = np.asarray(ecg_clean(ecg_signal_ecg, sampling_rate=fs))
        _, rpeaks = ecg_peaks(ecg_signal_processed, sampling_rate=fs)
        
        if rpeaks["ECG_R_Peaks"] is None or len(rpeaks["ECG_R_Peaks"]) < 8:
            print(f"Too few R-peaks detected {len(rpeaks['ECG_R_Peaks'])} in ECG signal, skipping processing.")
            return None, None
        
        mean_val = np.mean(ecg_signal_processed)
        std_val = np.std(ecg_signal_processed)
        ecg_signal_normalized = (ecg_signal_processed - mean_val) / (std_val + 1e-6)
        
    except Exception as e:
        print(f"Skipping ECG processing due to error: {e}")
        return None, None
    
    return ecg_signal_normalized, ecg_signal_processed

def preprocessSignalEDA(signal: pd.DataFrame, fs: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Function for preprocessing EDA signal and extracting phasic component using neurokit2. 
    The function also normalizes the signal using z-score.

    Args:
        signal (pandas DataFrame): _pandas DataFrame containing the EDA signal with a column named "gsr".
        fs (int): Sampling frequency of the EDA signal.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: The processed and normalized EDA signal and the raw EDA signal.
    """
    eda_signal_eda = signal["gsr"].to_numpy()
    
    if len(eda_signal_eda) < 1024:
        print("EDA signal is too short, skipping processing.")
        return None, None
    
    try:
        eda_signal_eda = eda_clean(eda_signal_eda, sampling_rate=fs)
        eda_signal_processed = eda_phasic(eda_signal_eda, sampling_rate=fs)["EDA_Phasic"].to_numpy()
        
        mean_val = np.mean(eda_signal_processed)
        std_val = np.std(eda_signal_processed)
        eda_signal_normalized = (eda_signal_processed - mean_val) / (std_val + 1e-6)
        
    except Exception as e:
        print(f"Skipping EDA processing due to error: {e}")
        return None, None
    
    return eda_signal_normalized, eda_signal_processed