import random
import numpy as np
import pandas as pd
import shap
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from torch.utils.data import DataLoader

from modules.gru_model import EnhancedGRUModel, load_model
from modules.cognitive_load_dataset_infer import CognitiveLoadDataset, evaluate_random_subset, load_data_subject_split, random_subject_windows
from modules.plot import create_prediction_viz, plot_signal_with_shap

def set_seed(seed=42):
    random.seed(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_shap_values(model, background_data, target_sample):
    model.eval()

    indices = np.random.choice(len(background_data), 100, replace=False)
    
    background = torch.tensor(background_data[indices], dtype=torch.float32)
    target = torch.tensor(target_sample, dtype=torch.float32)

    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(target)[0]

    return shap_values[0] if isinstance(shap_values, list) else shap_values

    
if "__main__" == __name__:
    set_seed(41)
    model_path = "models/final_model_86.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
    viz_data = random_subject_windows("datasets", n_subjects=5)
    create_prediction_viz(viz_data, loaded_model)
    
    test_dataset = CognitiveLoadDataset(X_all, y_all)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    evaluate_random_subset(loaded_model, test_dataset, n_samples=400)

    idx = np.random.randint(len(X_all))

    X_sample = X_all[idx][None, ...]
    shap_vals = compute_shap_values(loaded_model, X_all, X_sample)

    plot_signal_with_shap(
        X_sample=X_sample[0],
        shap_vals=shap_vals,
        title=f"SHAP interpretace – vzorek {idx}",
    )

    global_importance = np.mean(np.abs(shap_vals), axis=0)

    print(f"Global importance: {global_importance}")
    
