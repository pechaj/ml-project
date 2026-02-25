import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        attn_scores = self.attn(x) # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * x, dim=1)
        return context_vector, attn_weights

class EnhancedGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, bidirectional=True):
        super(EnhancedGRUModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Snížíme časové rozlišení (lepší pro GRU)
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        direction_factor = 2 if bidirectional else 1
        gru_input_dim = 64 # Odpovídá počtu filtrů v poslední CNN vrstvě

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3
        )

        self.gru_dim = hidden_size * direction_factor
        self.attention = Attention(self.gru_dim)

        # Klasifikační hlava
        self.fc = nn.Sequential(
            nn.Linear(self.gru_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.cnn(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)

        context_vector, _ = self.attention(gru_out)

        out = self.fc(context_vector)

        return out
    
    def predict(self, x, threshold=0.6):
        """
        Predikuje binární třídy pro vstupní tensor x.
        x: torch.Tensor ve tvaru (batch_size, window_size, channels) 
        nebo (batch_size, channels, window_size) podle tvé architektury.
        """
        self.eval() # Přepne model do eval módu (vypne Dropout apod.)
        with torch.no_grad():
            logits = self.forward(x).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int()
            
        return probs, preds

def sigmoid(x):
    """
    Sigmoid function for numpy arrays
    """
    return 1 / (1 + np.exp(-x))

def load_model(path, model, is_cuda=False):
    if is_cuda:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model
