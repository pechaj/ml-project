import random
import numpy as np
import pandas as pd
import os
import shap
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
from plotly.subplots import make_subplots
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from torch.utils.data import DataLoader

from modules.cognitive_load_dataset_infer import CognitiveLoadDataset, load_data_subject_split


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, bidirectional=True):
        super(GRUModel, self).__init__()

        direction_factor = 2 if bidirectional else 1
        self.gru_dim = hidden_size * direction_factor

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2,  # Přidán dropout mezi vrstvy GRU pro regulaci
        )

        self.layer_norm = nn.LayerNorm(self.gru_dim)

        # --- Zde přidáváme Attention vrstvu ---
        self.attention = Attention(self.gru_dim)

        self.fc1 = nn.Linear(self.gru_dim * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.layer_norm(gru_out)

        context_vector, weights = self.attention(
            gru_out
        )  # weights jsou (batch, seq_len, 1)
        last_hidden = gru_out[:, -1, :]  # get last time step

        combined = torch.cat((last_hidden, context_vector), dim=1)
        x = F.relu(self.fc1(combined))

        # x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        out = self.fc2(x)

        return out  # , weights # VRACÍME I VÁHY

    def predict(self, x, device="cpu"):
        model = self.to(device)
        self.eval()

        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in x:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()

                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        # Convert to numpy arrays
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)

        # Apply threshold to get binary predictions
        probabilities = sigmoid(all_outputs)
        predicted_labels = (probabilities > 0.6).astype(int)

        # Calculate metrics
        accuracy = (predicted_labels == all_labels).mean()
        f1 = f1_score(all_labels, predicted_labels)
        balanced_acc = balanced_accuracy_score(all_labels, predicted_labels)

        print("\nEvaluation metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")

        return all_labels, predicted_labels, probabilities


class GRUModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.sigmoid(out)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, gru_output):
        # gru_output tvar: (batch, seq_len, hidden_dim)

        # Spočítáme skóre pro každý časový krok
        attn_weights = self.attn(gru_output)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Vynásobíme výstupy GRU jejich vahami
        context_vector = torch.sum(attn_weights * gru_output, dim=1)

        return context_vector, attn_weights


def set_seed(seed=42):
    random.seed(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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

def plot_subject_predictions(
    X,
    y_true,
    y_pred,
    y_prob,
    groups,
    class_names=("Low", "High"),
):
    SIGNAL_COLOR_ECG = "rgba(232, 24, 24, 0.8)"
    SIGNAL_COLOR_EDA = "rgba(24, 156, 232, 0.8)"

    N, T, _ = X.shape
    correct = y_true == y_pred

    unique_subjects = list(dict.fromkeys(groups))

    frames = []

    for subject in unique_subjects:
        subject_indices = [i for i, g in enumerate(groups) if g == subject]

        traces, shapes, annotations = [], [], []
        offset = 0

        for idx, i in enumerate(subject_indices):
            t = np.arange(T) + offset

            traces += [
                go.Scatter(
                    x=t,
                    y=X[i, :, 0],
                    mode="lines",
                    line=dict(color=SIGNAL_COLOR_ECG),
                    name="ECG",
                    showlegend=(idx == 0),
                ),
                go.Scatter(
                    x=t,
                    y=X[i, :, 1],
                    mode="lines",
                    line=dict(color=SIGNAL_COLOR_EDA),
                    name="EDA",
                    showlegend=(idx == 0),
                ),
            ]

            # Green/red background for correct/wrong prediction
            bg = "rgba(0,200,0,0.12)" if correct[i] else "rgba(200,0,0,0.12)"
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=offset, x1=offset + T,
                y0=0, y1=1,
                fillcolor=bg, line=dict(width=0), layer="below",
            ))

            # Divider line between windows
            if idx < len(subject_indices) - 1:
                shapes.append(dict(
                    type="line", xref="x", yref="paper",
                    x0=offset + T, x1=offset + T,
                    y0=0, y1=1,
                    line=dict(color="rgba(0,0,0,0.4)", dash="dash", width=1),
                ))

            annotations.append(dict(
                x=offset + T / 2,
                y=1.10,
                xref="x", yref="paper",
                showarrow=False,
                font=dict(size=11),
                text=(
                    f"<b>{'✓ OK' if correct[i] else '✗ FAIL'}</b><br>"
                    f"GT: {class_names[y_true[i]]}<br>"
                    f"PRED: {class_names[y_pred[i]]} ({y_prob[i]:.2f})"
                ),
            ))

            offset += T

        frames.append(go.Frame(
            name=subject,
            data=traces,
            layout=dict(
                shapes=shapes,
                annotations=annotations,
                title_text=f"Subject: <b>{subject}</b> — {len(subject_indices)} consecutive windows",
                xaxis=dict(
                    range=[0, len(subject_indices) * T],
                    tickvals=[i * T + T // 2 for i in range(len(subject_indices))],
                    ticktext=[f"Win {i+1}" for i in range(len(subject_indices))],
                ),
            ),
        ))

    # --- Initial frame ---
    fig = go.Figure(data=frames[0].data, frames=frames)

    fig.update_layout(
        title=f"Subject: <b>{unique_subjects[0]}</b>",
        height=680,
        margin=dict(t=120),       # extra top space for annotations above plot
        legend=dict(x=0, y=1.18, orientation="h"),
        xaxis=dict(
            range=[0, len([i for i, g in enumerate(groups) if g == unique_subjects[0]]) * T],
            tickvals=[i * T + T // 2 for i in range(len([i for i, g in enumerate(groups) if g == unique_subjects[0]]))],
            ticktext=[f"Win {i+1}" for i in range(len([i for i, g in enumerate(groups) if g == unique_subjects[0]]))],
        ),
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Subject: ", font=dict(size=13)),
            pad=dict(t=50),
            steps=[
                dict(
                    method="animate",
                    args=[
                        [subject],
                        dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0)),
                    ],
                    label=subject,
                )
                for subject in unique_subjects
            ],
        )],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.18, x=1.0, xanchor="right",
            buttons=[
                dict(label="▶ Play through",
                     method="animate",
                     args=[None, dict(frame=dict(duration=1200), fromcurrent=True)]),
                dict(label="⏹ Stop",
                     method="animate",
                     args=[[None], dict(mode="immediate")]),
            ],
        )],
    )

    fig.show()

def compute_shap_values(model, background_data, target_sample):
    model.eval()

    background = background_data[:100]
    target = torch.tensor(target_sample, dtype=torch.float32)

    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(target)[0]

    return shap_values[0] if isinstance(shap_values, list) else shap_values


def plot_signal_with_shap(
    X_sample: np.ndarray, shap_vals: np.ndarray, title="SHAP Interpretace"
):
    shap_vals = np.squeeze(shap_vals)
    if len(X_sample.shape) == 3:
        X_sample = X_sample[0]

    channels = ["EKG (ECG)", "EDA"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"{ch} - Důležitost úseků" for ch in channels],
    )

    for i in range(2):
        sig = X_sample[:, i]
        s_val = shap_vals[:, i]

        window_size = 30 if i == 0 else 100

        s_val_smoothed = (
            pd.Series(s_val)
            .rolling(window=window_size, center=True)
            .mean()
            .fillna(0)
            .values
        )

        x_axis = np.arange(len(sig))
        v_limit = np.percentile(np.abs(s_val_smoothed), 99.5)

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=sig,
                mode="lines",
                line=dict(color="rgba(150, 150, 150, 0.5)", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=i + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=sig,
                mode="markers",
                marker=dict(
                    size=6,
                    color=s_val_smoothed,
                    colorscale="ylorrd",
                    cmid=0,
                    cmin=-v_limit,
                    cmax=v_limit,
                    colorbar=dict(
                        title="SHAP",
                        x=1.02,
                        len=0.45,
                        y=0.75 if i == 0 else 0.25,
                    )
                    if i == 0
                    else None,
                    showscale=True if i == 0 else False,
                ),
                text=[f"SHAP: {val:.4f}" for val in s_val_smoothed],
                name=channels[i],
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        height=800,
        width=1000,
        title_text=title,
        template="plotly_white",
        showlegend=False,
    )

    fig.update_xaxes(title_text="Vzorky (Samples)", row=2, col=1)
    fig.show()

    
if "__main__" == __name__:
    model_path = "models/cognitive_load_model_82_ds_64.pth"

    input_size = 2  # ECG and EDA channels
    hidden_size = 128
    output_size = 1
    batch_size = 64

    model = GRUModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        bidirectional=True,
    )

    print("Loading model from:", model_path)
    loaded_model = load_model(model_path, model, is_cuda=torch.cuda.is_available())

    if not loaded_model:
        print("Model loading failed.")

    print("Model loaded successfully.")

    X_train, X_test, y_train, y_test, groups_train, groups_test = load_data_subject_split("datasets")
    test_dataset = CognitiveLoadDataset(X_test, y_test)

    test_loader = DataLoader(test_dataset, batch_size=64)

    y_true, y_pred, y_prob = loaded_model.predict(test_loader, device="cpu")

    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)

    plot_subject_predictions(
        X=X_test,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=confidence,
        groups=groups_test,
        class_names=("Low", "High"),
    )

    idx = np.random.randint(len(X_test))

    X_sample = X_test[idx][None, ...]
    shap_vals = compute_shap_values(loaded_model, X_test, X_sample)

    plot_signal_with_shap(
        X_sample=X_sample[0],
        shap_vals=shap_vals,
        title=f"SHAP interpretace – vzorek {idx}",
    )

    global_importance = np.mean(np.abs(shap_vals), axis=0)

    print(f"Global importance: {global_importance}")
