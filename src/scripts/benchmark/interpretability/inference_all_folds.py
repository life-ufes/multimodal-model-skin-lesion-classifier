import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

# Ajuste conforme seu ambiente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from benchmark.models import multimodalIntraInterModal

# ==========================================================
# CONFIGURAÇÕES TÉCNICAS E FIXAS
# ==========================================================
LABELS = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
PAD_COLUMNS = ["patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age", "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic", "itch", "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"]
NUMERICAL_COLS = ["age", "diameter_1", "diameter_2"]
CATEGORICAL_COLS = ["smoke", "drink", "background_father", "background_mother", "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation"]

MISSING_RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caminhos de dados
METADATA_FILE = "./data/PAD-UFES-20/metadata.csv"
IMAGE_ROOT = "./data/PAD-UFES-20/images"
OHE_PATH = "./data/preprocess_data/ohe_pad_20.pickle"
SCALER_PATH = "./data/preprocess_data/scaler_pad_20.pickle"

# ==========================================================
# FUNÇÕES DE APOIO
# ==========================================================

def clean_metadata(df):
    df = df.fillna("EMPTY")
    df = df.replace([r"^\s*$", " ", "  ", "NÃO  ENCONTRADO"], "EMPTY", regex=True)
    df = df.replace("BRASIL", "BRAZIL")
    return df

def process_metadata(df, ohe, scaler):
    features = df.copy()
    features[CATEGORICAL_COLS] = features[CATEGORICAL_COLS].astype(str)
    features[NUMERICAL_COLS] = features[NUMERICAL_COLS].apply(pd.to_numeric, errors="coerce").fillna(-1)
    cat_data = ohe.transform(features[CATEGORICAL_COLS])
    num_data = scaler.transform(features[NUMERICAL_COLS])
    processed = np.hstack([cat_data, num_data])
    return torch.tensor(processed, dtype=torch.float32).to(DEVICE)

def apply_missing_metadata(sample_df, rate, seed):
    rng = random.Random(seed)
    cols = NUMERICAL_COLS + CATEGORICAL_COLS
    n_to_drop = int(round(len(cols) * rate))
    to_drop = rng.sample(cols, n_to_drop)
    for col in to_drop:
        if col in NUMERICAL_COLS: sample_df[col] = -1
        else: sample_df[col] = "EMPTY"
    return sample_df

# ==========================================================
# EXECUÇÃO PRINCIPAL
# ==========================================================

if __name__ == "__main__":
    with open(OHE_PATH, "rb") as f: shared_ohe = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: shared_scaler = pickle.load(f)
    
    transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    meta_all = pd.read_csv(METADATA_FILE)

    # Lista de backbones e mecanismos que você quer testar
    BACKBONES = ["caformer_b36.sail_in22k_ft_in1k"]
    MECHANISMS = [
        "att-intramodal+residual+cross-attention-metadados", 
        "crossattention", 
        "concatenation", 
        "metablock"
    ]

    for cnn_name in BACKBONES:
        for attention in MECHANISMS:
            print(f"\n{'='*60}\n⚙️ EXPERIMENTO: {cnn_name} | {attention}\n{'='*60}")
            
            # Ajuste o BASE_DIR conforme a estrutura de pastas do mecanismo
            # Note que mudei a parte final do path para 'attention'
            base_results_dir = f"./src/results/testes-da-implementacao-final_2/01012026/PAD-UFES-20/unfrozen_weights/8/{attention}/model_{cnn_name}_with_one-hot-encoder_512_with_best_architecture"
            
            all_folds_data = []

            for fold_idx in range(1, 6):
                fold_folder = f"{cnn_name}_fold_{fold_idx}"
                fold_path = os.path.join(base_results_dir, fold_folder)
                model_path = os.path.join(fold_path, "model.pth")
                preds_csv = os.path.join(fold_path, f"predictions_eval_fold_{fold_idx}.csv")

                if not os.path.exists(model_path):
                    print(f"⚠️ Pulo: {model_path} não existe.")
                    continue

                # Load Model
                model = multimodalIntraInterModal.MultimodalModel(
                    num_classes=6, device=DEVICE, cnn_model_name=cnn_name,
                    text_model_name="one-hot-encoder", vocab_size=91, num_heads=2,
                    attention_mecanism=attention, n=2, unfreeze_weights="unfrozen_weights"
                )
                ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
                state_dict = ckpt.get("model_state_dict", ckpt)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
                model.to(DEVICE).eval()

                preds_fold = pd.read_csv(preds_csv)
                df_test = pd.merge(meta_all, preds_fold, left_on="img_id", right_on="image_name")

                for rate in MISSING_RATES:
                    y_true, y_pred = [], []
                    for _, row in df_test.iterrows():
                        sample_df = pd.DataFrame([row[PAD_COLUMNS].to_dict()])
                        seed = hash((fold_idx, row["img_id"], rate)) % (2**32)
                        sample_missing = clean_metadata(apply_missing_metadata(sample_df, rate, seed))
                        
                        img = Image.open(os.path.join(IMAGE_ROOT, row["img_id"]))
                        img_t = transform(image=np.array(img.convert("RGB")))["image"].unsqueeze(0).to(DEVICE)
                        meta_t = process_metadata(sample_missing, shared_ohe, shared_scaler)

                        with torch.no_grad():
                            output = model(img_t, meta_t)
                            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                        
                        y_true.append(LABELS.index(row["diagnostic"]))
                        y_pred.append(np.argmax(probs))

                    bacc = balanced_accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average="weighted")
                    
                    all_folds_data.append({
                        "backbone": cnn_name,
                        "mechanism": attention,
                        "fold": fold_idx,
                        "missing_rate": rate,
                        "balanced_acc": bacc,
                        "f1_weighted": f1
                    })
                    print(f"   Fold {fold_idx} | Rate {rate:.1f} | BAcc: {bacc:.4f}")

            # --- PROCESSAMENTO ESTATÍSTICO (MÉDIA E DESVIO) ---
            if all_folds_data:
                df_results = pd.DataFrame(all_folds_data)
                
                # Agrupa por missing_rate e calcula estatísticas
                stats = df_results.groupby("missing_rate").agg({
                    "balanced_acc": ["mean", "std"],
                    "f1_weighted": ["mean", "std"]
                }).reset_index()
                
                # Achata os nomes das colunas
                stats.columns = ["missing_rate", "bacc_mean", "bacc_std", "f1_mean", "f1_std"]
                stats["backbone"] = cnn_name
                stats["mechanism"] = attention

                # Salva o log detalhado e o resumo estatístico
                output_path = f"./src/results/summary_{cnn_name}_{attention}.csv"
                stats.to_csv(output_path, index=False)
                print(f"📊 Resumo salvo em: {output_path}")

    print("\n✅ Todos os modelos foram processados!")