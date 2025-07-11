from modules.PreprocessData import preprocessDataset
from modules.ioPart import loadData, saveData
from modules.CognitiveLoadDataset import GRUModel, create_data_loaders, CNNStressClassifier, browse_windows, plot_signal_windows
import modules.CognitiveLoadDataset as cld
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

low_load_dir = "datasets/Low_load"
high_load_dir = "datasets/High_load"

# Ensure destination directories exist else preprocess the dataset
if (os.path.exists(low_load_dir) and os.path.exists(high_load_dir)):
    print("Directories already exist, no need to preprocess.")

else:
    os.makedirs(low_load_dir, exist_ok=True)
    os.makedirs(high_load_dir, exist_ok=True)
    # Create the directories
    for i in range(1, 60):
        filename = f"Part{i}"
        for j in range(1, 37):
            ecg_signal, eda_signal = loadData(filename, j)
            if ecg_signal is None or eda_signal is None:
                continue
            
            ecg_signal_filtered, eda_signal_filtered = preprocessDataset(
                ecg_signal, eda_signal, 256
            )
            if ecg_signal_filtered is None or eda_signal_filtered is None:
                continue
            
            saveData(ecg_signal_filtered, "ecg", low_load_dir, high_load_dir, filename, j)
            saveData(eda_signal_filtered, "eda", low_load_dir, high_load_dir, filename, j)

# Load dataset
base_dir = "./datasets"
print("Loading data from folders...")
# Enable class balancing at the dataset level
X, y = cld.load_data_from_folders(base_dir, balance_classes=True)
cld.plot_ecg_eda_window(X, y, 2806, 256)

# Define model hyperparameters
input_size = 2
hidden_size = 128
output_size = 1
learning_rate = 1e-3
batch_size = 32

# Split data into train and test sets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create balanced data loaders
class_counts = np.bincount(y)
total_samples = len(y)
class_weights = 1. / class_counts
sample_weights = class_weights[y]

print("Class counts:", class_counts)
print("Class weights:", class_weights)
print("Sample weights:", sample_weights[:10])

print("Splitting data into train and test sets...")
train_loader, test_loader = create_data_loaders(X, y, sample_weights, batch_size)

# Calculate class weights for weighted loss

print("Initializing model...")
model = GRUModel(input_size, hidden_size, output_size, True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

# Define model path
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelGRU_new.pth")

# Check data distribution before training
train_labels = train_loader.dataset.y.cpu().numpy()
print(f"\nClass distribution in training data: {np.unique(train_labels, return_counts=True)}")
plot_signal_windows(X, labels=y, class_to_plot=0, num_samples=5, title="EDA / ECG Windows")
plot_signal_windows(X, labels=y, class_to_plot=1, num_samples=5, title="EDA / ECG Windows")

# Plot 20 windows regardless of class
plot_signal_windows(X, num_samples=20, title="Random Signal Windows")
# Option 1: Train new model
# batch_x, batch_y = next(iter(train_loader))
# print(batch_x.shape, batch_y.shape)

# # Training loop
# for epoch in range(100):
#     model.train()
#     optimizer.zero_grad()
    
#     outputs = model(batch_x).squeeze(1)  # shape: (batch,)
#     loss = criterion(outputs, batch_y)
    
#     loss.backward()
#     optimizer.step()

#     # Metrics
#     with torch.no_grad():
#         preds = (torch.sigmoid(outputs) > 0.5).float()
#         acc = (preds == batch_y).float().mean().item()
#         print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc*100:.2f}%")
        
# for i in range(4):
#     ecg = batch_x[i, :, 0].cpu().numpy()  # ECG signal
#     eda = batch_x[i, :, 1].cpu().numpy()  # EDA signal
#     label = batch_y[i].item()

#     plt.figure(figsize=(12, 4))
#     plt.subplot(2, 1, 1)
#     plt.plot(ecg, label='ECG')
#     plt.title(f"Sample {i} - Label: {label}")
#     plt.legend()

#     plt.subplot(2, 1, 2)
#     plt.plot(eda, label='EDA', color='orange')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

cld.train_model(model, train_loader, criterion, optimizer, epochs=10)
cld.save_model(model, model_path)

# Option 2: Load existing model if needed
# model = cld.load_model(model_path, model)


# Evaluate the model and get predictions with optimal threshold
true_labels, predicted_labels = cld.evaluate_model(model, test_loader)

# Plot confusion matrix with improved predictions
print("\nPlotting confusion matrix...")
cld.plot_confusion_matrix(true_labels, predicted_labels)

# Plot example signals with both true and predicted labels
print("\nPlotting signal with predictions...")
# browse_windows(X, y_true=y, model=model, device='cuda' if torch.cuda.is_available() else 'cpu')
