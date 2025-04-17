import wfdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class WaveNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, residual_channels, num_blocks, num_layers
    ):
        super(WaveNet, self).__init__()
        self.causal_conv = nn.Conv1d(
            in_channels, residual_channels, kernel_size=2, padding=1, dilation=1
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(residual_channels, num_layers) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.causal_conv(x)[:, :, :-1]  # Ensure causality
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, num_layers):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    residual_channels,
                    residual_channels,
                    kernel_size=2,
                    dilation=2**i,
                    padding=2**i,
                )
                for i in range(num_layers)
            ]
        )
        self.skip_convs = nn.ModuleList(
            [
                nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        skip_out = torch.zeros_like(x)
        for conv, skip in zip(self.layers, self.skip_convs):
            out = conv(x)
            out = torch.tanh(out) * torch.sigmoid(out)
            
            skip_out_tmp = skip(out)
        
            # Ensure skip_out and skip_out_tmp have the same size
            if skip_out_tmp.size(2) > skip_out.size(2):
                skip_out_tmp = skip_out_tmp[:, :, :skip_out.size(2)]  # Crop skip_out_tmp if it's too large
            elif skip_out_tmp.size(2) < skip_out.size(2):
                skip_out_tmp = F.pad(skip_out_tmp, (0, skip_out.size(2) - skip_out_tmp.size(2)))  # Pad skip_out_tmp if it's too small

            # Add the skip connection
            skip_out += skip_out_tmp
            
            if out.size(2) > x.size(2):
                out = out[:, :, :x.size(2)]  # Crop out if it's too large
            elif x.size(2) > out.size(2):
                out = F.pad(out, (0, x.size(2) - out.size(2)))  # Pad out if it's too small
            x = out + x  # Residual connection
        return skip_out


def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for noisy, clean in dataloader:
            optimizer.zero_grad()
            # print(f"Shape of noisy tensor: {noisy.shape}")
            # print(f"Shape of clean tensor: {clean.shape}")
            prediction = model(noisy)
            
            # print(f"Prediction shape: {prediction.shape}")
            # print(f"Clean shape: {clean.shape}")
            
            clean = clean[:, :, :prediction.shape[2]]
            
            loss = criterion(prediction, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")


def add_noise(ecg, noise_type="gaussian", snr_db=20):
    if noise_type == "gaussian":
        signal_power = np.mean(ecg**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(ecg))
        return ecg + noise
    elif noise_type == "baseline_wander":
        baseline = 0.5 * np.sin(2 * np.pi * 0.1 * np.arange(len(ecg)) / 250)
        return ecg + baseline


def segment_signal(signal, segment_length=500):
    segments = []
    for i in range(0, len(signal) - segment_length, segment_length):
        segments.append(signal[i : i + segment_length])
    return np.array(segments)


record = wfdb.rdrecord("datasets/mit-bih-arrhythmia-database-1.0.0/102")
signal = record.p_signal[:, 0]
signal = resample(signal, int(len(signal) * 250 / record.fs))

noisy_signal = add_noise(signal, noise_type="gaussian")

clean_segments = segment_signal(signal)
noisy_segments = segment_signal(noisy_signal)
clean_segments = (clean_segments - np.mean(clean_segments)) / np.std(clean_segments)
noisy_segments = (noisy_segments - np.mean(noisy_segments)) / np.std(noisy_segments)

criterion = nn.MSELoss()

clean_segments = torch.Tensor(clean_segments).unsqueeze(1)
noisy_segments = torch.Tensor(noisy_segments).unsqueeze(1)

# Update the WaveNet model's input channels
model = WaveNet(in_channels=1, out_channels=1, residual_channels=32, num_blocks=3, num_layers=4)

# Train the model
train_dataset = TensorDataset(torch.Tensor(noisy_segments), torch.Tensor(clean_segments))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, optimizer, criterion, epochs=20)

# Test and plot results
with torch.no_grad():
    test_noisy = noisy_segments[:1]
    test_clean = clean_segments[:1]
    prediction = model(test_noisy).squeeze(1).numpy()  # Remove channel dimension

plt.plot(test_noisy[0, 0].numpy(), label='Noisy')  # First batch, first channel
plt.plot(test_clean[0, 0].numpy(), label='Clean')  # First batch, first channel
plt.plot(prediction[0], label='Predicted')         # Predicted output
plt.legend()
plt.show()