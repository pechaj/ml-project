#!/usr/bin/env python
# coding: utf-8

# # Cognitive Load Classification from Physiological Signals
# 
# This notebook demonstrates processing and classification of cognitive load using physiological signals (ECG and EDA).

# ## Setup and Imports

# In[ ]:

# !pip install neurokit2 matplotlib numpy pandas scikit-learn torch tqdm

# In[ ]:

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ## Define Dataset and Data Processing Functions

# In[ ]:

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
    """
    Loads ECG and EDA data from specified directory structure.
    Implements window segmentation with 50% overlap.
    
    Args:
        base_dir: Directory containing High_load and Low_load folders
        balance_classes: Whether to compute sample weights for class balancing
        
    Returns:
        X: Feature array
        y: Labels array
        sample_weights: Optional weights for balanced sampling
    """
    X, y = [], []
    window_size = 256 * 20  # ~20 seconds at 256 Hz
    step_size = window_size // 2  # 50% overlap
    
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
    
    # Optionally compute sample weights for balancing classes
    if balance_classes:
        class_counts = np.bincount(y)
        class_weights = 1. / class_counts
        sample_weights = class_weights[y]
        print(f"Class weights for balancing: Class 0: {class_weights[0]:.4f}, Class 1: {class_weights[1]:.4f}")
        return X, y, sample_weights
    
    return X, y

def create_data_loaders(X, y, sample_weights=None, batch_size=32, test_split=0.2):
    """
    Create balanced data loaders using sample weights
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

def plot_ecg_eda_window(X, y, index=0, sampling_rate=256):
    """
    Plot ECG and EDA signals from a specific window
    """
    sample = X[index]
    window_size = sample.shape[0] // 2
    
    # Get ECG and EDA segments
    ecg = sample[:window_size, 0]
    eda = sample[:window_size, 1]
    
    time = np.linspace(0, window_size / sampling_rate, window_size)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, ecg, label='ECG', color='red')
    plt.title(f'ECG Signal (label: {y[index]})')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time, eda, label='EDA', color='blue')
    plt.title(f'EDA Signal (label: {y[index]})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_signal_windows(X, labels=None, class_to_plot=None, num_samples=5, title="Signal Windows"):
    """
    Plot multiple signal windows for visualization
    
    Args:
        X: Input data array
        labels: Optional labels array
        class_to_plot: If provided, only plot windows from this class
        num_samples: Number of windows to plot
        title: Plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Select indices to plot
    if labels is not None and class_to_plot is not None:
        # Get indices of samples from the specified class
        class_indices = np.where(labels == class_to_plot)[0]
        # Randomly select from those indices
        if len(class_indices) > num_samples:
            indices_to_plot = np.random.choice(class_indices, num_samples, replace=False)
        else:
            indices_to_plot = class_indices
    else:
        # Randomly select from all samples
        indices_to_plot = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    
    for i, idx in enumerate(indices_to_plot):
        sample = X[idx]
        window_size = sample.shape[0] // 2
        
        # Extract ECG and EDA
        ecg = sample[:window_size, 0]
        eda = sample[:window_size, 1]
        
        # Plot ECG
        plt.subplot(num_samples, 2, 2*i+1)
        plt.plot(ecg, 'r-')
        label_text = f"Window {idx}"
        if labels is not None:
            label_text += f" (Class {labels[idx]})"
        plt.title(f"ECG {label_text}")
        
        # Plot EDA
        plt.subplot(num_samples, 2, 2*i+2)
        plt.plot(eda, 'b-')
        plt.title(f"EDA {label_text}")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# ## Define Models

# In[ ]:

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
        self.fc1 = nn.Linear(hidden_size * direction_factor, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, output_size)

        self.apply(self._init_weights)

    def forward(self, x):
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden*dir)
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
            nn.Conv1d(1, 30, kernel_size=5, stride=1, padding=2),  # output: (30, input_length)
            nn.ReLU(),
            nn.MaxPool1d(2),  # (30, input_length/2)

            nn.Conv1d(30, 30, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # (30, input_length/4)

            nn.Conv1d(30, 60, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # (60, input_length/8)

            nn.Conv1d(60, 60, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)   # (60, input_length/16)
        )

        # BiGRU layer
        self.bigru = nn.GRU(
            input_size=60,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 40),  # BiGRU hidden size * 2 directions
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(40, num_classes)
        )

    def forward(self, x):
        # Input shape: [batch_size, sequence_length, channels]
        # CNN expects: [batch_size, channels, sequence_length]
        x = x.permute(0, 2, 1)
        
        # Apply CNN
        x = self.cnn(x)  
        
        # Prepare for GRU: [batch_size, sequence_length, channels]
        x = x.permute(0, 2, 1)
        
        # Apply BiGRU
        gru_out, _ = self.bigru(x)
        last_step = gru_out[:, -1, :]  # Take last time step
        
        # Apply classifier
        out = self.classifier(last_step)
        return out  # BCEWithLogitsLoss expects raw logits

# ## Training and Evaluation Functions

# In[ ]:

def train_model(model, train_loader, criterion, optimizer, epochs=10, device='cpu'):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        device: Device to use for training
        
    Returns:
        Trained model
    """
    model = model.to(device)
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
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # Check prediction distribution periodically
        if epoch % 5 == 0 or epoch == epochs - 1:
            check_prediction_distribution(model, train_loader, device)
    
    return model

def check_prediction_distribution(model, data_loader, device='cpu'):
    """
    Check the distribution of predictions
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)

    unique_preds, pred_counts = np.unique(np.array(all_preds), return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
    model.train()

def evaluate_model(model, test_loader, threshold=0.5, device='cpu'):
    """
    Evaluate the model
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        threshold: Classification threshold
        device: Device to use for evaluation
        
    Returns:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
    """
    model = model.to(device)
    model.eval()
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    # Convert to numpy arrays
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # Apply threshold to get binary predictions
    predicted_labels = (sigmoid(all_outputs) > threshold).astype(int)
    
    # Calculate metrics
    accuracy = (predicted_labels == all_labels).mean()
    f1 = f1_score(all_labels, predicted_labels)
    balanced_acc = balanced_accuracy_score(all_labels, predicted_labels)
    
    print(f"\nEvaluation metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    return all_labels, predicted_labels

def sigmoid(x):
    """
    Sigmoid function for numpy arrays
    """
    return 1 / (1 + np.exp(-x))

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Load", "High Load"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall/Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")

def save_model(model, path):
    """
    Save model to disk
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path, model):
    """
    Load model from disk
    """
    model.load_state_dict(torch.load(path))
    return model

# ## Google Drive Integration

# In[ ]:

# Uncomment to mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# ## Example Usage

# In[ ]:

# Helper functions for loading pre-processed data from Drive
def download_data():
    """
    Download example data or use existing data
    """
    # Check if data exists
    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)
        os.makedirs("datasets/Low_load", exist_ok=True)
        os.makedirs("datasets/High_load", exist_ok=True)
        
        print("Please upload your ECG/EDA data files to the datasets directory")
        print("The structure should be:")
        print("- datasets/Low_load/")
        print("- datasets/High_load/")
        print("With CSV files in the format: subject_block_signal.csv")
        print("Example: 1_4_ecg.csv, 1_4_eda.csv")
        return False
    
    return True

# Main execution
if __name__ == "__main__":
    # Check for data
    if download_data():
        # Load dataset
        base_dir = "./datasets"
        print("Loading data from folders...")
        X, y, sample_weights = load_data_from_folders(base_dir, balance_classes=True)
        
        # Plot some examples
        plot_ecg_eda_window(X, y, index=0, sampling_rate=256)
        plot_signal_windows(X, labels=y, class_to_plot=0, num_samples=3, title="Low Cognitive Load Samples")
        plot_signal_windows(X, labels=y, class_to_plot=1, num_samples=3, title="High Cognitive Load Samples")
        
        # Define model hyperparameters
        input_size = 2  # ECG and EDA channels
        hidden_size = 128
        output_size = 1
        learning_rate = 1e-3
        batch_size = 32
        
        # Create balanced data loaders
        train_loader, test_loader = create_data_loaders(X, y, sample_weights, batch_size)
        
        # Initialize model
        print("Initializing model...")
        model = GRUModel(input_size, hidden_size, output_size, bidirectional=True).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), learning_rate)
        
        # Define model path
        model_path = "cognitive_load_model.pth"
        
        # Train the model
        model = train_model(model, train_loader, criterion, optimizer, epochs=10, device=device)
        
        # Save the model
        save_model(model, model_path)
        
        # Evaluate the model
        true_labels, predicted_labels = evaluate_model(model, test_loader, device=device)
        
        # Plot confusion matrix
        plot_confusion_matrix(true_labels, predicted_labels)
