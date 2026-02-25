import random
import numpy as np
import pandas as pd
import shap
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from torch.utils.data import DataLoader

from modules.gru_model import EnhancedGRUModel, load_model
from modules.cognitive_load_dataset_infer import CognitiveLoadDataset, evaluate_random_subset, load_data_subject_split
from modules.plot import create_interactive_slider_viz

def set_seed(seed=42):
    random.seed(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
    set_seed(43)
    model_path = "models/final_model_.pth"

    input_size = 2  # ECG and EDA channels
    hidden_size = 128
    output_size = 1
    batch_size = 64

    model = EnhancedGRUModel(
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

    X_all, y_all, groups_all = load_data_subject_split("datasets")
    
    create_interactive_slider_viz(X_all, y_all, groups_all, loaded_model, n_random_subjects=5)
    
    test_dataset = CognitiveLoadDataset(X_all, y_all)

    test_loader = DataLoader(test_dataset, batch_size=64)
    
    evaluate_random_subset(loaded_model, test_dataset, n_samples=200)
"""
    y_true, y_pred, y_prob = loaded_model.predict(test_loader, device="cpu")

    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)

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
    """
