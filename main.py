from modules.PreprocessData import preprocessDataset
from modules.ioPart import loadData, saveData
from modules.CognitiveLoadDataset import CognitiveLoadDataset, GRUModel
from modules.CognitiveLoadDataset import train_model, evaluate_model,load_data_from_folders
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

low_load_dir = "datasets/CLAS_database/Low_load"
high_load_dir = "datasets/CLAS_database/High_load"

torch.cuda.is_available()

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
            
            t, ecg_signal_filtered, eda_signal_filtered = preprocessDataset(ecg_signal, eda_signal)
            if ecg_signal_filtered is None or eda_signal_filtered is None:
                continue
            
            saveData(ecg_signal_filtered, "ecg", low_load_dir, high_load_dir, filename, j)
            saveData(eda_signal_filtered, "eda", low_load_dir, high_load_dir, filename, j)

# Load dataset
base_dir = "./datasets/CLAS_database"
print("Loading data from folders...")
X, y = load_data_from_folders(base_dir)

# Split data into train and test sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create PyTorch dataloaders
train_dataset = CognitiveLoadDataset(X_train, y_train)
test_dataset = CognitiveLoadDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss, and optimizer
print("Initializing model...")
input_size = 2
hidden_size = 64
output_size = 1
model = GRUModel(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)    

# Train the model
print("Training model...")
train_model(model, train_loader, criterion, optimizer)

# Evaluate the model
evaluate_model(model, test_loader)