import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score

# Custom dataset class
class CognitiveLoadDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data_from_folders(base_dir, balance_classes=True):
    X, y = [], []
    window_size = 256 * 20  # 30 seconds at 256 Hz
    step_size = window_size // 2  # 50% overlap
    windows = []
    indices = []
    
    class_samples = {0: [], 1: []}  # For storing samples by class
    
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
            recordings[subject_key][signal_type] = pd.read_csv(file_path)
        
        for subject_key, signals in recordings.items():
            if 'ecg' in signals and 'eda' in signals:
                ecg = signals['ecg']
                eda = signals['eda']
                
                min_len = min(len(ecg), len(eda))
                ecg, eda = ecg[:min_len], eda[:min_len]
                
                for start in range(0, min_len - window_size + 1, step_size):
                    ecg_window = ecg[start:start + window_size]
                    eda_window = eda[start:start + window_size]
                    combined_signal = pd.concat([ecg_window, eda_window], axis=1)
                    # Store samples by class for balancing
                    class_samples[label].append(combined_signal)
                    end = start + window_size
                    windows.append(ecg[start:end])
                    indices.append((start, end))
    
    # Combine data
    for label in [0, 1]:
        for sample in class_samples[label]:
            X.append(sample)
            y.append(label)
    
    # Print original class distribution
    class_0_size = y.count(0)
    class_1_size = y.count(1)
    print(f"Original class distribution - Class 0: {class_0_size}, Class 1: {class_1_size}")
    
    # Convert list of DataFrames to a 3D numpy array
    X = np.stack([sample.values for sample in X])
    y = np.array(y)
    
    return X, y

def create_data_loaders(X, y, sample_weights=None, batch_size=32, test_split=0.2):
    """
    Create balanced data loaders using the sample weights
    """

    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=test_split, random_state=42, stratify=y
    )
    
    # Create datasets
    train_dataset = CognitiveLoadDataset(X_train, y_train)
    test_dataset = CognitiveLoadDataset(X_test, y_test)
    
    # Create data loaders
    if sample_weights is not None:
        # Get sample weights for training samples
        train_sample_weights = sample_weights[idx_train]
        
        # Create a sampler
        train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

def plot_ecg_eda_window(X, y, index=0, sampling_rate=1000):
    
    sample = X[index]
    window_size = sample.shape[0] // 2
    
    # Rozdělení EKG a EDA
    ecg = sample[:window_size, 0]
    eda = sample[:window_size, 1]
    
    time = np.linspace(0, window_size / sampling_rate, window_size)

    plt.figure(figsize=(12, 5))

    plt.plot(time, ecg, label='EKG', color='red')
    plt.title(f'EKG signál (label: {y[index]})')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.plot(time, eda, label='EDA', color='blue')
    plt.title(f'EDA signál (label: {y[index]})')
    plt.xlabel('Čas (s)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, bidirectional=True):
        super(GRUModel, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Update the input size of the fully connected layer
        direction_factor = 2 if bidirectional else 1
        self.layer_norm = nn.LayerNorm(hidden_size * direction_factor)
        self.fc1 = nn.Linear(hidden_size * direction_factor, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

        self.apply(self._init_weights)

    def forward(self, x):
        # print(x)
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden*dir)
        gru_out = self.layer_norm(gru_out)
        last_hidden = gru_out[:, -1, :]  # get last time step
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        out = self.fc2(x)
        return out  # raw logits

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

class CNN_BiGRU_Model(nn.Module):
    def __init__(self, input_length=1280, num_classes=1):
        super(CNN_BiGRU_Model, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=5, stride=1, padding=2),  # output: (30, 1280)
            nn.ReLU(),
            nn.MaxPool1d(2),  # (30, 640)

            nn.Conv1d(30, 30, kernel_size=5, stride=1, padding=2),  # (30, 640)
            nn.ReLU(),
            nn.MaxPool1d(2),  # (30, 320)

            nn.Conv1d(30, 60, kernel_size=3, stride=1, padding=1),  # (60, 320)
            nn.ReLU(),
            nn.MaxPool1d(2),  # (60, 160)

            nn.Conv1d(60, 60, kernel_size=3, stride=1, padding=1),  # (60, 160)
            nn.ReLU(),
            nn.MaxPool1d(2)   # (60, 80)
        )

        # BiGRU (input size matches CNN output channels)
        self.bigru = nn.GRU(
            input_size=60,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 40),  # BiGRU hidden size * 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(40, num_classes)  # Output: 1 for binary classification
        )

    def forward(self, x):
        # x: [batch_size, 1, 1280]
        x = self.cnn(x)  # Output: [batch_size, 60, 80]
        x = x.permute(0, 2, 1)  # [batch_size, 80, 60] -> sequence_len=80

        # GRU expects (batch, seq_len, feature_dim)
        gru_out, _ = self.bigru(x)  # Output: [batch_size, 80, 128]
        last_step = gru_out[:, -1, :]  # Take last time step
        out = self.classifier(last_step)
        return out  # BCEWithLogitsLoss expects raw logits


# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    writer = SummaryWriter()
    model.train()
    
    # Calculate class balance in training set
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.extend(y_batch.cpu().numpy())
    
    unique_labels, label_counts = np.unique(np.array(all_labels), return_counts=True)
    print(f"Training data class distribution: {dict(zip(unique_labels, label_counts))}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        
        total_loss = 0
        correct, total = 0, 0
        
        for X_batch, y_batch in epoch_iter:
            y_batch = y_batch.view(-1).float()
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()  # Ensures shape [batch_size]
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            

        # Calculate and log training metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total * 100

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # Check prediction distribution periodically
        if epoch % 10 == 0 or epoch == epochs - 1:
            check_prediction_distribution(model, train_loader)
    
    writer.flush()
    writer.close()
    
def check_prediction_distribution(model, data_loader, device='cpu'):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            outputs = model(X_batch).squeeze()
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())

    unique_preds, pred_counts = np.unique(np.array(all_preds), return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
    model.train()

# Evaluation function
def evaluate_model(model, test_loader, threshold=0.5):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    raw_outputs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            raw_outputs.extend(outputs.cpu().numpy())
            predicted = (outputs > threshold).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    # Find optimal threshold based on predictions
    print("Analyzing model outputs to find optimal threshold...")
    raw_outputs = np.array(raw_outputs)
    all_labels_np = np.array(all_labels)
    
    # Try different thresholds to find the best one
    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.1, 0.9, 0.05):
        new_preds = (raw_outputs > t).astype(float)
        f1 = f1_score(all_labels_np, new_preds)
        bal_acc = balanced_accuracy_score(all_labels_np, new_preds)
        print(f"Threshold {t:.2f} - F1: {f1:.4f}, Balanced Acc: {bal_acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    # Recalculate predictions with optimal threshold
    print(f"\nUsing optimal threshold: {best_threshold:.2f}")
    all_preds = (raw_outputs > best_threshold).astype(float)
    accuracy = np.mean(all_preds == all_labels_np)
    print(f"Test Accuracy with optimal threshold: {accuracy * 100:.2f}%")
        # Print prediction distribution
    unique_labels, label_counts = np.unique(np.array(all_labels), return_counts=True)
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    print(f"True label distribution: {dict(zip(unique_labels, label_counts))}")
    print(f"Predicted label distribution: {dict(zip(unique_preds, pred_counts))}")
    
    return all_labels_np, all_preds
    
def save_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_model(path, model):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model
    
def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix for model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for the classes. Default is ["Low Load", "High Load"]
    """
    if class_names is None:
        class_names = ["Low_load", "High_load"]
        
    # Convert to numpy arrays if they're torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Print raw values to see what we're working with
    print(f"Sample of true values: {y_true[:10]}")
    print(f"Sample of predicted values: {y_pred[:10]}")
    
    # Ensure values are integers to avoid confusion matrix issues
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
        
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix raw values:\n{cm}")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.show()
    
def browse_windows(X, y_true, model, device='cpu'):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import numpy as np
    import torch

    fig, ax = plt.subplots(figsize=(12, 3))
    plt.subplots_adjust(bottom=0.2)

    index = [0]

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_tensor).squeeze().cpu().numpy()
        pred_labels = (preds >= 0.5).astype(int)

    lines = []
    colors = ['black', 'orange', 'blue', 'green']
    n_channels = X.shape[2] if X.ndim == 3 else 1

    if n_channels == 1:
        line, = ax.plot(X[0], color='black')
        lines = [line]
    else:
        for ch in range(n_channels):
            l, = ax.plot(X[0, :, ch], color=colors[ch % len(colors)], label=f'Channel {ch}')
            lines.append(l)
        ax.legend()

    title = ax.set_title(f'Window 0 | True: {y_true[0]} | Pred: {pred_labels[0]}')

    def update_plot(i):
        for ch, line in enumerate(lines):
            if n_channels == 1:
                line.set_ydata(X[i])
            else:
                line.set_ydata(X[i, :, ch])
        ax.relim()
        ax.autoscale_view()
        title.set_text(f'Window {i} | True: {y_true[i]} | Pred: {pred_labels[i]}')
        fig.canvas.draw_idle()

    def next_window(event):
        if index[0] < len(X) - 1:
            index[0] += 1
            update_plot(index[0])

    def prev_window(event):
        if index[0] > 0:
            index[0] -= 1
            update_plot(index[0])

    axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.21, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Previous')
    bnext.on_clicked(next_window)
    bprev.on_clicked(prev_window)

    update_plot(0)
    plt.show()

def plot_signal_windows(windows, labels=None, class_to_plot=None, num_samples=20, show_mean=True, title="EDA Signal Overlay"):
    """
    Plots overlaid EDA windows (from channel 1) for visualization.

    Args:
        windows (np.ndarray): Shape (num_windows, seq_len, channels). EDA must be at channel index 1.
        labels (np.ndarray or list, optional): Corresponding labels for each window.
        class_to_plot (int or None): If given, filter to only windows with this label.
        num_samples (int): Number of signal windows to overlay.
        show_mean (bool): Whether to also plot the mean signal across selected windows.
        title (str): Plot title.
    """
    plt.figure(figsize=(12, 6))

    # Convert to array if not already
    windows = np.array(windows)

    if windows.ndim != 3 or windows.shape[2] <= 1:
        raise ValueError("Expected input shape (num_windows, seq_len, channels) with EDA in channel 1.")
    
    # Extract EDA channel (assumed at index 1)
    eda_signals = windows[:, :, 1]

    # Filter by class if specified
    if labels is not None and class_to_plot is not None:
        indices = [i for i, lbl in enumerate(labels) if lbl == class_to_plot]
        if not indices:
            print(f"No windows found for class {class_to_plot}")
            return
        eda_signals = eda_signals[indices]

    # Limit number of samples
    n = min(num_samples, len(eda_signals))
    selected = eda_signals[:n]

    # Plot individual signals
    for i in range(n):
        plt.plot(selected[i], alpha=0.4, linewidth=1)

    # Optionally plot mean
    if show_mean:
        mean_signal = np.mean(selected, axis=0)
        plt.plot(mean_signal, color='black', linewidth=2.5, label='Mean')

    label_text = f" - Class {class_to_plot}" if class_to_plot is not None else ""
    plt.title(f"{title}{label_text}")
    plt.xlabel("Time steps")
    plt.ylabel("EDA signal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
