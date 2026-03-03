import os
import random
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import DataLoader, Dataset, Subset

class CognitiveLoadDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_data_subject_split(base_dir, target_fs=128):
    """
    Loads ECG and EDA data.
    Splits dataset on SUBJECT level to avoid leakage.
    Then creates 50% overlapping windows.
    Returns also groups (subject_id) for LOSO.
    """
    target_fs = 128
    orig_fs = 256
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
            ecg = signals['ecg'].iloc[:, 0].values.astype(np.float32)
            eda = signals['eda'].iloc[:, 0].values.astype(np.float32)

            ecg_ds = downsample_signal(ecg, orig_fs, target_fs)
            eda_ds = downsample_signal(eda, orig_fs, target_fs)

            final_min_len = min(len(ecg_ds), len(eda_ds))

            for start in range(0, final_min_len - window_size + 1, step_size):
                ecg_w = ecg_ds[start : start + window_size]
                eda_w = eda_ds[start : start + window_size]
                
                combined = np.stack([ecg_w, eda_w], axis=1)
                X.append(combined)
                y.append(label)
                groups.append(subject_id)
        

        return np.array(X), np.array(y), np.array(groups)

    X, y, groups = create_windows(all_recordings)

    return X, y, groups

def downsample_signal(signal, orig_fs, target_fs):
    gcd = np.gcd(orig_fs, target_fs)
    up = target_fs // gcd
    down = orig_fs // gcd

    return resample_poly(signal, up, down)

def evaluate_random_subset(model, dataset, n_samples=200, threshold=0.6, device='cuda' if torch.cuda.is_available() else 'cpu'):
    num_total = len(dataset)
    indices = np.random.choice(num_total, min(n_samples, num_total), replace=False)
    
    subset = Subset(dataset, indices) # type: ignore
    subset_loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    model.to(device)
    model.eval()
    
    all_true, all_preds = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in subset_loader:
            X_batch = X_batch.to(device)
            
            outputs = model(X_batch).view(-1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).float()
            
            all_true.extend(y_batch.numpy())
            all_preds.extend(preds.cpu().numpy())
            
    print(f"\n✨ RYCHLÁ STATISTIKA ({n_samples} náhodných oken) ✨")
    print(f"{'='*40}")
    print(f"Accuracy: {accuracy_score(all_true, all_preds):.2%}")
    print(f"{'='*40}")
    print(classification_report(all_true, all_preds, target_names=["Low", "High"]))
    
def random_subject_windows(base_dir, n_subjects=3, target_fs=128):
    orig_fs = 256
    window_size = target_fs * 20
    step_size = window_size // 2
    
    # Cesta k surovým datům
    raw_base_dir = os.path.join(base_dir, "Raw")
    
    # Najdeme dostupné subjekty v normalizovaných složkách
    low_files = os.listdir(os.path.join(base_dir, "Low_load"))
    high_files = os.listdir(os.path.join(base_dir, "High_load"))
    all_subjects = sorted(list(set([f.split('_')[0] for f in low_files + high_files if '_' in f])))
    
    selected_sids = random.sample(all_subjects, min(n_subjects, len(all_subjects)))
    viz_data = []

    for sid in selected_sids:
        # 1. Náhodný výběr třídy (pro model/okna)
        label = random.choice([0, 1])
        class_name = "Low_load" if label == 0 else "High_load"
        norm_class_dir = os.path.join(base_dir, class_name)
        
        # Najdeme soubor v normalizované složce
        subject_files = [f for f in os.listdir(norm_class_dir) if f.startswith(sid) and 'ecg' in f.lower()]
        if not subject_files:
            continue
        
        ecg_file_name = random.choice(subject_files)
        eda_file_name = ecg_file_name.replace('ecg', 'eda').replace('ECG', 'EDA')

        # 2. NAČTENÍ NORMALIZOVANÝCH DAT (pro model)
        ecg_norm = pd.read_csv(os.path.join(norm_class_dir, ecg_file_name)).iloc[:, 0].values
        eda_norm = pd.read_csv(os.path.join(norm_class_dir, eda_file_name)).iloc[:, 0].values
        
        # 3. NAČTENÍ RAW DAT (pro graf)
        raw_class_dir = os.path.join(raw_base_dir, class_name)
        ecg_raw = pd.read_csv(os.path.join(raw_class_dir, ecg_file_name)).iloc[:, 0].values
        eda_raw = pd.read_csv(os.path.join(raw_class_dir, eda_file_name)).iloc[:, 0].values

        # 4. DOWNSAMPLING (provádíme na obou sadách stejně, aby seděl čas)
        # Modelová data (znormalizovaná)
        ecg_norm_ds = downsample_signal(ecg_norm, orig_fs, target_fs)
        eda_norm_ds = downsample_signal(eda_norm, orig_fs, target_fs)
        
        # Vizualizační data (surová)
        ecg_raw_ds = downsample_signal(ecg_raw, orig_fs, target_fs)
        eda_raw_ds = downsample_signal(eda_raw, orig_fs, target_fs)
        
        # Sjednocení délek
        min_l = min(len(ecg_norm_ds), len(ecg_raw_ds), len(eda_norm_ds), len(eda_raw_ds))
        ecg_n, eda_n = ecg_norm_ds[:min_l], eda_norm_ds[:min_l]
        ecg_r, eda_r = ecg_raw_ds[:min_l], eda_raw_ds[:min_l]

        # 5. TVORBA OKEN PRO MODEL (z normalizovaných dat)
        X_windows = []
        for start in range(0, min_l - window_size + 1, step_size):
            win = np.stack([ecg_n[start:start+window_size], 
                            eda_n[start:start+window_size]], axis=1)
            X_windows.append(win)
        
        viz_data.append({
            'subject_id': sid,
            'label': label,
            'full_ecg': ecg_r,  # Do grafu jdou RAW data
            'full_eda': eda_r,  # Do grafu jdou RAW data
            'X_windows': np.array(X_windows) # Do modelu jdou NORMALIZOVANÁ okna
        })
        
    return viz_data