import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ==========================================================
# PATH SETUP
# ==========================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from models import multimodalIntraInterModal # Certifique-height que o path está correto

# ==========================================================
# CONFIGURAÇÕES E PATHS
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "./data/PAD-UFES-20"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
ENCODER_DIR = "./data/preprocess_data"

BASE_RESULTS_DIR = "./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/unfrozen_weights/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture"

OUT_DIR = "./results/flip_analysis_complete"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_LIST = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
FEATURE_LIST = ["age","region","gender","itch","grew","bleed","changed","elevation","hurt"]
NUMERICAL_COLS = ["age","diameter_1","diameter_2"]
DROP_COLS = ["patient_id","lesion_id","img_id","biopsed","diagnostic"]
K = len(CLASS_LIST)

# ==========================================================
# CARREGAMENTO DE RECURSOS (OHE, SCALER, DF)
# ==========================================================
df_all_meta = pd.read_csv(METADATA_PATH)

with open(os.path.join(ENCODER_DIR,"ohe_pad_20.pickle"),"rb") as f:
    OHE = pickle.load(f)

with open(os.path.join(ENCODER_DIR,"scaler_pad_20.pickle"),"rb") as f:
    SCALER = pickle.load(f)

# ==========================================================
# FUNÇÕES DE PROCESSAMENTO
# ==========================================================
def load_image_transform(size=(224,224)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def process_image(path):
    img = Image.open(path).convert("RGB")
    tensor = load_image_transform()(img).unsqueeze(0).to(DEVICE)
    return tensor

def process_metadata_row(row_dict):
    df = pd.DataFrame([row_dict])
    features = df.drop(columns=DROP_COLS, errors='ignore')

    categorical_cols = [c for c in features.columns if c not in NUMERICAL_COLS]
    features[categorical_cols] = features[categorical_cols].astype(str)
    features[NUMERICAL_COLS] = features[NUMERICAL_COLS].apply(pd.to_numeric, errors="coerce").fillna(-1)

    cat = OHE.transform(features[categorical_cols])
    num = SCALER.transform(features[NUMERICAL_COLS])
    processed = np.hstack([cat, num])
    return torch.tensor(processed, dtype=torch.float32).to(DEVICE)

def load_model(model_path, unfreeze_weights="unfrozen_weights"):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=K, device=DEVICE, cnn_model_name="densenet169",
        text_model_name="one-hot-encoder", vocab_size=91, num_heads=8,
        attention_mecanism="gfcam", n=2, unfreeze_weights=unfreeze_weights
    )
    ckpt = torch.load(model_path, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model

# ==========================================================
# MUTAÇÃO E PREDIÇÃO
# ==========================================================
def mutate_metadata(row, feature):
    r = copy.deepcopy(row)
    # Toggles booleanos/binários (seguro para strings do CSV)
    if feature in ["itch","grew","bleed","changed","hurt","elevation"]:
        val = str(r[feature]).lower()
        r[feature] = "False" if val in ["true", "1.0", "1"] else "True"
    
    elif feature == "age":
        r[feature] = 80 # Simula paciente idoso
    
    elif feature == "gender":
        r[feature] = "MALE" if str(r[feature]).upper() == "FEMALE" else "FEMALE"
    
    elif feature == "region":
        r[feature] = "FACE" if str(r[feature]).upper() != "FACE" else "FOREARM"
    
    return r

def predict_class(model, image_tensor, metadata_tensor):
    with torch.no_grad():
        logits = model(image_tensor, metadata_tensor)
        return int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())

# ==========================================================
# LOOP PRINCIPAL POR FOLD
# ==========================================================
def run_fold(fold_number, unfreeze_weights):
    print(f"\n>>> Iniciando Fold {fold_number}...")
    fold_dir = os.path.join(BASE_RESULTS_DIR, f"densenet169_fold_{fold_number}")
    model = load_model(os.path.join(fold_dir, "model.pth"), unfreeze_weights)
    
    df_val = pd.read_csv(os.path.join(fold_dir, f"predictions_eval_fold_{fold_number}.csv"))
    df_meta = df_all_meta[df_all_meta["img_id"].isin(df_val["image_name"].tolist())]

    fold_metrics = {
        "flips": {f: 0 for f in FEATURE_LIST},
        "y_true": [], "y_pred": [],
        "transitions": {f: {"c0": [], "c1": []} for f in FEATURE_LIST},
        "total_samples": 0
    }

    for _, row in df_meta.iterrows():
        img_path = os.path.join(IMAGE_DIR, row["img_id"])
        if not os.path.exists(img_path): continue

        img_t = process_image(img_path)
        meta_dict = row.to_dict()
        
        # Predição Original (Baseline)
        c0 = predict_class(model, img_t, process_metadata_row(meta_dict))
        
        fold_metrics["y_true"].append(CLASS_LIST.index(row["diagnostic"]))
        fold_metrics["y_pred"].append(c0)
        fold_metrics["total_samples"] += 1

        # Mutações
        for f in FEATURE_LIST:
            mutated_dict = mutate_metadata(meta_dict, f)
            c1 = predict_class(model, img_t, process_metadata_row(mutated_dict))
            
            fold_metrics["transitions"][f]["c0"].append(c0)
            fold_metrics["transitions"][f]["c1"].append(c1)
            if c1 != c0:
                fold_metrics["flips"][f] += 1

    # Calcula taxas para o fold
    fold_rates = {f: fold_metrics["flips"][f] / fold_metrics["total_samples"] for f in FEATURE_LIST}
    return fold_rates, fold_metrics

def main():
    unfreeze_weights = "unfrozen_weights"
    all_fold_rates = []
    global_y_true, global_y_pred = [], []
    global_trans = {f: {"c0": [], "c1": []} for f in FEATURE_LIST}

    for f_idx in range(1, 6):
        rates, metrics = run_fold(f_idx, unfreeze_weights)
        all_fold_rates.append(rates)
        global_y_true.extend(metrics["y_true"])
        global_y_pred.extend(metrics["y_pred"])
        for f in FEATURE_LIST:
            global_trans[f]["c0"].extend(metrics["transitions"][f]["c0"])
            global_trans[f]["c1"].extend(metrics["transitions"][f]["c1"])

    # --- VISUALIZAÇÃO 1: FLIP RATE (Média e Desvio) ---
    df_rates = pd.DataFrame(all_fold_rates)
    summary = pd.DataFrame({
        "feature": df_rates.columns,
        "mean": df_rates.mean(),
        "std": df_rates.std()
    }).sort_values("mean", ascending=False)
    
    summary.to_csv(os.path.join(OUT_DIR, "flip_summary.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(summary["feature"], summary["mean"], xerr=summary["std"], capsize=5, color='skyblue', edgecolor='black')
    plt.xlabel("Mean Flip Rate (Impacto)")
    plt.title(f"Sensibilidade do Modelo por Feature (5 Folds)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_flip_rates.png"), dpi=300)

    # --- VISUALIZAÇÃO 2: MATRIZ DE CONFUSÃO GLOBAL ---
    cm_global = confusion_matrix(global_y_true, global_y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_global, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LIST, yticklabels=CLASS_LIST)
    plt.title("Matriz de Confusão Agregada (Base)")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_global_confusion.png"), dpi=300)

    # --- VISUALIZAÇÃO 3: MATRIZES DE TRANSIÇÃO ---
    for f in FEATURE_LIST:
        tm = confusion_matrix(global_trans[f]["c0"], global_trans[f]["c1"], labels=range(K))
        plt.figure(figsize=(9, 7))
        sns.heatmap(tm, annot=True, fmt='d', cmap='YlOrRd', xticklabels=CLASS_LIST, yticklabels=CLASS_LIST)
        plt.title(f"Transição de Classe: Efeito de '{f}'")
        plt.ylabel("Predição Original ($C_0$)")
        plt.xlabel("Predição Pós-Mutação ($C_1$)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"transition_{f}.png"), dpi=300)
        plt.close()

    print(f"\nSucesso! Resultados salvos em: {OUT_DIR}")

if __name__ == "__main__":
    main()