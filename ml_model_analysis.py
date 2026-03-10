import random
import numpy as np
import shap
from sklearn.isotonic import spearmanr
import torch
from torch.utils.data import DataLoader

from modules.gru_model import EnhancedGRUModel, load_model
from modules.cognitive_load_dataset_infer import CognitiveLoadDataset, evaluate_random_subset, load_data_subject_split, random_subject_windows
from modules.plot import create_prediction_viz, plot_signal_with_shap, plot_correlation_distribution

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

    shap_values = explainer.shap_values(target)
    
    if isinstance(shap_values, (list, tuple)):
        shap_values = np.array(shap_values[0])
    else:
        shap_values = np.array(shap_values)
    
    if len(shap_values.shape) == 3:
        shap_values = shap_values[0]

    return shap_values

def compute_global_importance_stats(model, X_data, n_samples=100):
    """
    Vypočítá průměrnou SHAP důležitost napříč více náhodnými vzorky.
    """
    model.eval()
    all_importances = []
    
    print(f"Spouštím hromadnou analýzu SHAP pro {n_samples} vzorků...")
    
    # Náhodný výběr indexů pro analýzu
    indices = np.random.choice(len(X_data), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Příprava vzorku (přidání dimenze batch)
        X_sample = X_data[idx][None, ...]
        
        # Výpočet SHAP hodnot (využijeme vaši stávající funkci)
        shap_vals = compute_shap_values(model, X_data, X_sample)
        
        # Vyčištění dimenzí (z (1, 2560, 2, 1) na (2560, 2))
        shap_cleaned = np.squeeze(shap_vals)
        if shap_cleaned.ndim != 2:
            shap_cleaned = shap_vals[0, :, :, 0]
            
        # Výpočet absolutní důležitosti pro dané okno (průměr přes čas)
        # Výsledek je pole [imp_EKG, imp_EDA]
        window_importance = np.mean(np.abs(shap_cleaned), axis=0)
        all_importances.append(window_importance)
        
        if (i + 1) % 10 == 0:
            print(f"Zpracováno {i + 1}/{n_samples} vzorků...")

    # Převod na numpy array pro snadnou agregaci
    all_importances = np.array(all_importances)
    
    # Finální průměr přes všechna vybraná okna
    global_mean_importance = np.mean(all_importances, axis=0)
    
    # Výpočet procentuálních vah
    total = np.sum(global_mean_importance)
    weights_pct = (global_mean_importance / total) * 100
    
    # Výpočet variability (směrodatná odchylka) - skvělé do diplomky!
    std_importance = np.std(all_importances, axis=0)
    
    print("\n" + "="*40)
    print(f"FINÁLNÍ GLOBÁLNÍ METRIKY (n={n_samples})")
    print("="*40)
    print(f"EKG váha: {global_mean_importance[0].item():.6f} ({weights_pct[0].item():.2f} % ± {np.std(all_importances[:,0]/np.sum(all_importances, axis=1)*100):.2f} %)")
    print(f"EDA váha: {global_mean_importance[1].item():.6f} ({weights_pct[1].item():.2f} % ± {np.std(all_importances[:,1]/np.sum(all_importances, axis=1)*100):.2f} %)")
    print("="*40 + "\n")
    
    return global_mean_importance, weights_pct

def compute_global_correlation_stats(model, X_data, n_samples=50):
    model.eval()
    correlations = []
    
    print(f"Spouštím hromadnou analýzu korelací pro {n_samples} vzorků...")
    indices = np.random.choice(len(X_data), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        X_sample = X_data[idx][None, ...]
        shap_vals = compute_shap_values(model, X_data, X_sample)
        
        # Squeeze na (2560, 2)
        shap_cleaned = np.squeeze(shap_vals)
        if shap_cleaned.ndim != 2:
            shap_cleaned = shap_vals[0, :, :, 0]
            
        # Spearmanova korelace pro toto konkrétní okno
        corr, _ = spearmanr(shap_cleaned[:, 0], shap_cleaned[:, 1])
        correlations.append(corr)
        
        if (i + 1) % 10 == 0:
            print(f"Zpracováno {i + 1}/{n_samples} korelací...")

    correlations = np.array(correlations)
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    
    print("\n" + "="*40)
    print("STATISTIKA ZÁVISLOSTI (Spearmanova korelace)")
    print("="*40)
    print(f"Průměrná korelace: {mean_corr:.4f}")
    print(f"Směrodatná odchylka: {std_corr:.4f}")
    print(f"Rozsah: {correlations.min():.4f} až {correlations.max():.4f}")
    print("="*40 + "\n")
    
    return correlations


if "__main__" == __name__:
    set_seed(48)
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

    #global_imp, global_weights = compute_global_importance_stats(loaded_model, X_all, n_samples=50)

    # Poté můžete stále vykreslit jeden konkrétní zajímavý graf pro vizuální ukázku
    idx = np.random.randint(len(X_all))
    X_sample = X_all[idx][None, ...]
    shap_vals = compute_shap_values(loaded_model, X_all, X_sample)
    shap_vals = np.squeeze(shap_vals)

    plot_signal_with_shap(
        X_sample=X_sample[0],
        shap_vals=shap_vals,
        title=f"SHAP interpretace – vzorek {idx}",
    )
    
    correlations = compute_global_correlation_stats(loaded_model, X_all, n_samples=50)
    plot_correlation_distribution(correlations)
    
