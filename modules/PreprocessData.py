import numpy as np
import pandas as pd
import random
from neurokit2 import ecg_process, ecg_clean, eda_phasic

def preprocessDataset(ecg_signal_full, eda_signal_full, fs):
    
    # Extract ECG signal and remove NaN values
    ecg_signal_full = ecg_signal_full.dropna()
    eda_signal_full = eda_signal_full.dropna()
    
    ecg_signal_filtered = preprocessSignalECG(ecg_signal_full, fs)
    eda_signal_filtered = preprocessSignalEDA(eda_signal_full, fs)
    
    return ecg_signal_filtered, eda_signal_filtered
    
def preprocessSignalECG(signal, fs):
    
    if "ecg2" not in signal.columns:
            raise ValueError("Column 'ecg2' not found in the dataset.")
        
    ecg_signal_ecg = signal["ecg2"][:].astype(float).values 
    
    if len(ecg_signal_ecg) < 1024:
        print("ECG signal is too short, skipping processing.")
        return None, None
    
    # ecg_signal_time = np.arange(0, len(ecg_signal_ecg)) / fs
    
    try:
        # Try to process the ECG signal
        ecg_signal_processed = ecg_clean(ecg_signal_ecg, sampling_rate=fs)
    except Exception as e:
        print(f"Skipping ECG processing due to error: {e}")
        return None, None
    
    return ecg_signal_processed

def preprocessSignalEDA(signal, fs):
    eda_signal_eda = signal["gsr"][:].astype(float).values 
    
    if len(eda_signal_eda) < 1024:
        return None, None
    
    try:
        eda_signal_processed = eda_phasic(eda_signal_eda, sampling_rate=fs)["EDA_Phasic"]
    except Exception as e:
        print(f"Skipping EDA processing due to error: {e}")
        return None, None
    
    
    return eda_signal_processed