from modules.preprocess_module import preprocessDataset
from modules.load_save_module import loadData, saveData
import os
# import matplotlib.pyplot as plt

low_load_dir = "datasets/Low_load"
high_load_dir = "datasets/High_load"

low_load_raw_dir = "datasets/Raw/Low_load"
high_load_raw_dir = "datasets/Raw/High_load"

# Ensure destination directories exist else preprocess the dataset
if os.path.exists(low_load_dir) and os.path.exists(high_load_dir):
    print("Directories already exist, no need to preprocess.")

else:
    os.makedirs(low_load_dir, exist_ok=True)
    os.makedirs(high_load_dir, exist_ok=True)
    os.makedirs(low_load_raw_dir, exist_ok=True)
    os.makedirs(high_load_raw_dir, exist_ok=True)
    
    # Create the directories
    for i in range(1, 61):
        filename = f"Part{i}"
        for j in range(1, 37):
            ecg_signal, eda_signal = loadData(filename, j)
            
            if ecg_signal is None or eda_signal is None:
                continue

            ecg_signal_filtered, eda_signal_filtered, ecg_signal_raw, eda_signal_raw = preprocessDataset(
                ecg_signal, eda_signal, 256
            )
            
            if ecg_signal_filtered is None or eda_signal_filtered is None:
                continue
            
            """plt.style.use("ggplot")
            
            fig, ax = plt.subplots()
            
            ax.plot(ecg_signal_filtered, color="blue")
            ax.plot(eda_signal_filtered, color="orange")
            plt.show()"""
            
            if ecg_signal_filtered is None or eda_signal_filtered is None:
                continue

            saveData(
                ecg_signal_filtered, "ecg", low_load_dir, high_load_dir, filename, j
            )
            saveData(
                ecg_signal_raw, "ecg", low_load_raw_dir, high_load_raw_dir, filename, j
            )
            
            saveData(
                eda_signal_filtered, "eda", low_load_dir, high_load_dir, filename, j
            )
            saveData(
                eda_signal_raw, "eda", low_load_raw_dir, high_load_raw_dir, filename, j
            )

""" # Load dataset
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
class_weights = 1.0 / class_counts
sample_weights = class_weights[y]

print("Class counts:", class_counts)
print("Class weights:", class_weights)
print("Sample weights:", sample_weights[:10])

print("Splitting data into train and test sets...")
train_loader, test_loader = create_data_loaders(X, y, sample_weights, batch_size)

# Plot 20 windows regardless of class
plot_signal_windows(X, num_samples=20, title="Random Signal Windows") """