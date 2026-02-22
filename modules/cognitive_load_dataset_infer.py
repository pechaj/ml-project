import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset, DataLoader

class CognitiveLoadDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_data_subject_split(base_dir, test_size=0.2, random_state=42):
    """
    Loads ECG and EDA data.
    Splits dataset on SUBJECT level to avoid leakage.
    Then creates 50% overlapping windows.
    """
    target_fs = 128

    window_size = target_fs * 20
    step_size = window_size // 2

    all_recordings = []

    print(f"🚀 Načítám data z: {base_dir}")

    # -------------------------
    # 1️⃣ Nejprve načteme metadata o všech subjektech
    # -------------------------
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
                all_recordings.append((subject_key, data["signals"], data["label"]))

    print(f"📊 Celkem subjektů/záznamů: {len(all_recordings)}")

    # -------------------------
    # 2️⃣ Split na úrovni subjektu
    # -------------------------
    subject_ids = [rec[0] for rec in all_recordings]
    labels = [rec[2] for rec in all_recordings]

    train_ids, test_ids = train_test_split(
        subject_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    print(f"✂️ Train subjektů: {len(train_ids)} | Test subjektů: {len(test_ids)}")

    # -------------------------
    # 3️⃣ Funkce pro tvorbu oken
    # -------------------------
    def create_windows(recordings_subset):
        X, y, groups = [], [], []

        for subject_key, signals, label in recordings_subset:

            ecg = signals['ecg']
            eda = signals['eda']

            orig_fs = 256

            initial_min_len = min(len(ecg), len(eda))
            ecg_data_orig = ecg.iloc[:initial_min_len, 0].values
            eda_data_orig = eda.iloc[:initial_min_len, 0].values

            # Downsampling
            ecg_data_ds = downsample_signal(ecg_data_orig, orig_fs, target_fs)
            eda_data_ds = downsample_signal(eda_data_orig, orig_fs, target_fs)

            final_min_len = min(len(ecg_data_ds), len(eda_data_ds))
            ecg_data = ecg_data_ds[:final_min_len]
            eda_data = eda_data_ds[:final_min_len]

            for start in range(0, final_min_len - window_size + 1, step_size):
                ecg_w = ecg_data[start:start + window_size]
                eda_w = eda_data[start:start + window_size]

                combined = np.stack([ecg_w, eda_w], axis=1)

                X.append(combined)
                y.append(label)
                groups.append(subject_key)  # <- přidáno

        return np.array(X), np.array(y), groups

    # -------------------------
    # 4️⃣ Vytvoření train/test
    # -------------------------
    train_recordings = [rec for rec in all_recordings if rec[0] in train_ids]
    test_recordings = [rec for rec in all_recordings if rec[0] in test_ids]

    X_train, y_train, groups_train = create_windows(train_recordings)
    X_test, y_test, groups_test = create_windows(test_recordings)

    print(f"\n📐 Train shape: {X_train.shape}")
    print(f"📐 Test shape:  {X_test.shape}")

    return X_train, X_test, y_train, y_test, groups_train, groups_test

def downsample_signal(signal, orig_fs, target_fs):
    if orig_fs == target_fs:
        return signal

    gcd = np.gcd(orig_fs, target_fs)
    up = target_fs // gcd
    down = orig_fs // gcd

    return resample_poly(signal, up, down)
