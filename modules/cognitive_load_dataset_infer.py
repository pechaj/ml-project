import os
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset

class CognitiveLoadDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_data_subject_split(base_dir):
    """
    Loads ECG and EDA data.
    Splits dataset on SUBJECT level to avoid leakage.
    Then creates 50% overlapping windows.
    Returns also groups (subject_id) for LOSO.
    """
    target_fs = 128
    window_size = target_fs * 20
    step_size = window_size // 2
    all_recordings = []

    print(f"🚀 Načítám data z: {base_dir}")

    for label, class_name in enumerate(["Low_load", "High_load"]):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        recordings = {}
        files = os.listdir(class_dir)

        for file in files:
            parts = file.rsplit('_', 2)
            if len(parts) != 3:
                continue

            subject_id, rec_num, signal_type = parts
            subject_key = f"{subject_id}_{rec_num}"
            signal_type = signal_type.replace('.csv', '').lower()

            if subject_key not in recordings:
                recordings[subject_key] = {"label": label, "signals": {}}

            file_path = os.path.join(class_dir, file)
            recordings[subject_key]["signals"][signal_type] = pd.read_csv(file_path)

        for subject_key, data in recordings.items():
            if 'ecg' in data["signals"] and 'eda' in data["signals"]:
              subject_id = subject_key.split('_')[0]
              all_recordings.append((subject_id, data["signals"], data["label"]))

    num_of_subjects = np.unique([rec[0] for rec in all_recordings])

    print(f"📊 Celkem subjektů: {len(num_of_subjects)}")
    print(f"📊 Celkem bloků: {len(all_recordings)}")

    def create_windows(recordings_subset):
        X, y, groups = [], [], []
        for subject_id, signals, label in recordings_subset:
            ecg = signals['ecg']
            eda = signals['eda']

            orig_fs = 256

            min_len = min(len(ecg), len(eda))
            ecg_data_ds = downsample_signal(ecg.iloc[:min_len, 0].values, orig_fs, target_fs)
            eda_data_ds = downsample_signal(eda.iloc[:min_len, 0].values, orig_fs, target_fs)
            final_min_len = min(len(ecg_data_ds), len(eda_data_ds))

            for start in range(0, final_min_len - window_size + 1, step_size):
                ecg_w = ecg_data_ds[start:start + window_size]
                eda_w = eda_data_ds[start:start + window_size]
                combined = np.stack([ecg_w, eda_w], axis=1)
                X.append(combined)
                y.append(label)
                groups.append(subject_id)

        return np.array(X), np.array(y), np.array(groups)

    X, y, groups = create_windows(all_recordings)

    print(f"X type: {type(X)}, X shape: {getattr(X, 'shape', None)}")

    return X, y, groups

def downsample_signal(signal, orig_fs, target_fs):
    gcd = np.gcd(orig_fs, target_fs)
    up = target_fs // gcd
    down = orig_fs // gcd

    return resample_poly(signal, up, down)

def get_subjects_data(X, y, groups, target_subjects):
    """
    X: numpy array (okna, délka, kanály)
    y: numpy array (labely)
    groups: numpy array (subject_ids)
    target_subjects: list ID subjektů, které chceme (např. ['S01', 'S05'])
    """
    mask = np.isin(groups, target_subjects)
    return X[mask], y[mask], groups[mask]