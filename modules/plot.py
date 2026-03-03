import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch

    
def create_prediction_viz(viz_data, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    fs = 128
    step_t = 1280 / fs
    win_t = 2560 / fs

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.2, row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=("Surové fyziologické signály (Raw)", "Klasifikace (Zelená = Správně, Červená = Chyba)"),
        x_title="Čas [s]"
    )

    all_steps = []
    total_traces = 0

    for data in viz_data:
        sid = data['subject_id']
        y_true = data['label']
        label_text = "High Load" if y_true == 1 else "Low Load"
        
        # Příprava dat pro model
        X_tensor = torch.tensor(data['X_windows'], dtype=torch.float32).to(device)
        
        # Tohle není potřeba
        # if X_tensor.shape[2] == 2:
            # X_tensor = X_tensor.transpose(1, 2)
        
        model.eval()
        with torch.no_grad():
            probs, preds = model.predict(X_tensor, threshold=0.6)

        time_axis = np.arange(len(data['full_ecg'])) / fs
        
        # --- SIGNÁLY ---
        fig.add_trace(go.Scattergl(x=time_axis, y=data['full_ecg'], name="EKG",
                                   line=dict(color='firebrick', width=1), visible=False), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scattergl(x=time_axis, y=data['full_eda'], name="EDA",
                                   line=dict(color='royalblue', width=1.5), visible=False), row=1, col=1, secondary_y=True)

        n_win = len(preds)
        for i in range(n_win):
            t0, t1 = i * step_t, i * step_t + win_t
            is_correct = (y_true == preds[i])
            
            # Barva: Zelená vs Červená
            base_color = "0, 200, 100" if is_correct else "255, 0, 0"
            
            # Grafické oddělení překryvu (Sudá/Lichá okna)
            y_val = [0, 0, 1, 1, 0] if i % 2 == 0 else [0, 0, 0.6, 0.6, 0]
            opacity = 0.3 if i % 2 == 0 else 0.6
            
            fig.add_trace(go.Scatter(
                x=[t0, t1, t1, t0, t0], 
                y=y_val,
                fill="toself", 
                fillcolor=f"rgba({base_color}, {opacity})",
                line=dict(width=0.5, color="rgba(0,0,0,0.2)"),
                name=f"Win {i}",
                visible=False,
                showlegend=False, # Aby to nespamovalo legendu
                hoveron="fills", # DŮLEŽITÉ: hover reaguje na celou plochu krabice
                text=(f"<b>Okno {i}</b> ({t0:.0f}-{t1:.0f}s)<br>"
                      f"Skutečnost: {label_text}<br>"
                      f"Predikce: {'High' if preds[i]==1 else 'Low'}<br>"
                      f"Jistota: {probs[i]:.2f}"),
                hoverinfo="text"
            ), row=2, col=1)

        traces_for_sid = 2 + n_win
        all_steps.append((total_traces, traces_for_sid, sid, label_text))
        total_traces += traces_for_sid

    # --- LOGIKA SLIDERU ---
    steps = []
    for start, length, sid, l_text in all_steps:
        visibility_mask = [False] * total_traces
        visibility_mask[start : start + length] = [True] * length
        
        steps.append(dict(
            method="update",
            label=f"{sid} ({l_text})", # Info o třídě přímo na slideru
            args=[{"visible": visibility_mask}, {"title": f"Subjekt {sid} | Skutečný stav: {l_text}"}]
        ))

    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Vybrán: "}, steps=steps)],
        height=850,
        template="plotly_white",
        yaxis=dict(title="EKG [mV]"),
        yaxis2=dict(title="EDA [μS]", overlaying="y", side="right"),
        yaxis3=dict(range=[0, 1.1], showticklabels=False, fixedrange=True)
    )

    if all_steps:
        s, m, _, _ = all_steps[0]
        for i in range(s, s+m): 
            fig.data[i].visible = True

    fig.show()
    
def plot_signal_with_shap(
    X_sample: np.ndarray, shap_vals: np.ndarray, title="SHAP Interpretace"
):
    shap_vals = np.squeeze(shap_vals)
    if len(X_sample.shape) == 3:
        X_sample = X_sample[0]

    channels = ["EKG", "EDA"]

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