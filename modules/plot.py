import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import torch

from modules.gru_model import EnhancedGRUModel

def create_interactive_slider_viz(X: np.ndarray[tuple[int, int], np.dtype[np.float64]], y: np.ndarray[tuple[int], np.dtype[np.int64]], groups, model: EnhancedGRUModel, n_random_subjects=5, n_windows=10, fs=128):
    unique_subjects = np.unique(groups)
    selected_subjects = np.random.choice(list(unique_subjects), min(n_random_subjects, len(unique_subjects)), replace=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    window_size = X.shape[1]
    step_size = window_size // 2
    win_t = window_size / fs
    step_t = step_size / fs

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.75, 0.25],
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=("Signály (EKG + EDA)", "Klasifikace s 50% překryvem"),
        x_title="Čas [s]",
        y_title="Amplituda [mV]"
    )

    # 1. Pevné vertikální čáry (Grid) - vykreslíme jen jednou, slider je neovládá
    for i in range(n_windows + 1):
        end_t = (i * step_size + window_size) / fs
        fig.add_vline(x=end_t, line_dash="dot", line_color="rgba(150,150,150,0.3)", row=1, col=1) # type: ignore

    all_steps = []
    current_trace_idx = 0

    for sid in selected_subjects:
        mask = (groups == sid)
        X_sub = X[mask][:n_windows]
        y_sub = y[mask][:n_windows]
        
        # Predikce
        X_tensor = torch.tensor(X_sub, dtype=torch.float32).to(device)
        probs, preds = model.predict(X_tensor, threshold=0.6)

        # Rekonstrukce signálu
        ecg_vals, eda_vals = [], []
        for i in range(len(X_sub)):
            portion = step_size if i < len(X_sub) - 1 else window_size
            ecg_vals.extend(X_sub[i, :portion, 0])
            eda_vals.extend(X_sub[i, :portion, 1])
        time_axis = np.arange(len(ecg_vals)) / fs

        # --- PŘIDÁVÁNÍ STOP (TRACES) ---
        # DŮLEŽITÉ: Každý subjekt si nese vlastní EKG a EDA stopy
        fig.add_trace(go.Scattergl(x=time_axis, y=ecg_vals, name=f"ECG {sid}", line=dict(color='firebrick', width=1.5), visible=False), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scattergl(x=time_axis, y=eda_vals, name=f"EDA {sid}", line=dict(color='royalblue', width=2), visible=False), row=1, col=1, secondary_y=True)
        
        # Počet stop pro klasifikaci (krabice)
        box_traces_count = 0
        
        # Vykreslení krabic ve dvou průchodech pro správné vrstvení (Sada A vespod, Sada B nahoře)
        # Průchod 1: Sada A (Vysoké, i % 2 == 0)
        for i in range(len(y_sub)):
            if i % 2 != 0:
                continue
            t0, t1 = i * step_t, i * step_t + win_t
            yt, yp = y_sub[i], preds[i]
            color = f"rgba({('0,200,100' if yp==0 else '255,165,0') if yt==yp else '255,0,0'}, 0.4)"
            
            fig.add_trace(go.Scatter(x=[t0, t1, t1, t0, t0], y=[0, 0, 1, 1, 0], fill="toself", fillcolor=color, 
                                     line=dict(width=1, color="rgba(0,0,0,0.2)"), visible=False, showlegend=False,
                                     hoverinfo="text", text=f"{t0}s - {t1}s | Okno {i}<br>True: {yt}, Pred: {yp}<br>Conf: {probs[i]:.2f}"), row=2, col=1)
            box_traces_count += 1

        # Průchod 2: Sada B (Nižší, i % 2 != 0)
        for i in range(len(y_sub)):
            if i % 2 == 0:
                continue
            t0, t1 = i * step_t, i * step_t + win_t
            yt, yp = y_sub[i], preds[i]
            color = f"rgba({('0,200,100' if yp==0 else '255,165,0') if yt==yp else '255,0,0'}, 0.8)"
            
            fig.add_trace(go.Scatter(x=[t0, t1, t1, t0, t0], y=[0, 0, 0.6, 0.6, 0], fill="toself", fillcolor=color, 
                                     line=dict(width=1, color="rgba(0,0,0,0.2)"), visible=False, showlegend=False,
                                     hoverinfo="text", text=f"{t0}s - {t1}s | Okno {i}<br>True: {yt}, Pred: {yp}<br>Conf: {probs[i]:.2f}"), row=2, col=1)
            box_traces_count += 1

        total_subject_traces = 2 + box_traces_count # 2 jsou ECG + EDA
        all_steps.append((current_trace_idx, total_subject_traces, sid))
        current_trace_idx += total_subject_traces

    # --- SLIDER ---
    steps = []
    for start, length, sid in all_steps:
        vis = [False] * current_trace_idx
        vis[start : start + length] = [True] * length
        steps.append(dict(method="update", label=str(sid), args=[{"visible": vis}, {"title": f"Subjekt {sid}"}]))

    # Inicializace prvního subjektu
    if all_steps:
        for j in range(all_steps[0][1]): fig.data[j].visible = True

    fig.update_layout(sliders=[dict(active=0, steps=steps, pad={"t": 50})], height=750, template="plotly_white")
    fig.update_yaxes(range=[0, 1.1], showticklabels=False, row=2, col=1)
    fig.show()