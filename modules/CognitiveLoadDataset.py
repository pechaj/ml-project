import os
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Custom dataset class
class CognitiveLoadDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data_from_folders(base_dir):
    X, y = [], []
    scaler = StandardScaler()
    window_size = 1024
    step_size = 512
    
    for label, class_name in enumerate(["Low_load", "High_load"]):
        class_dir = os.path.join(base_dir, class_name)
        recordings = {}
        
        for file in os.listdir(class_dir):
            parts = file.rsplit('_', 2)
            if len(parts) != 3:
                continue  # Skip unexpected filenames
            subject_id, rec_num, signal_type = parts
            subject_key = f"{subject_id}_{rec_num}"
            signal_type = signal_type.replace('.csv', '').lower()
            
            if subject_key not in recordings:
                recordings[subject_key] = {}
            
            file_path = os.path.join(class_dir, file)
            recordings[subject_key][signal_type] = pd.read_csv(file_path).values
        
        for subject_key, signals in recordings.items():
            if 'ecg' in signals and 'eda' in signals:
                ecg = signals['ecg']
                eda = signals['eda']
                
                min_len = min(len(ecg), len(eda))
                ecg, eda = ecg[:min_len], eda[:min_len]
                
                for start in range(0, min_len - window_size + 1, step_size):
                    ecg_window = ecg[start:start + window_size]
                    eda_window = eda[start:start + window_size]
                    combined_signal = np.hstack((ecg_window, eda_window))
                    X.append(combined_signal)
                    y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    X = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
    return X, y

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        _, h = self.gru(x)
        x = self.fc1(h[-1])
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)
    

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")