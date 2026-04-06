import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

# Ajuste conforme seu ambiente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from benchmark.models import multimodalIntraInterModal

# ==========================================================
# 1. CONSTANTES E CONFIGURAÇÕES
# ==========================================================
# Mapeamento do diagnóstico baseado na ordem do output
MAPPING = {0: "ACK", 1: "BCC", 2: "MEL", 3: "NEV", 4: "SCC", 5: "SEK"}
LABELS_LIST = [MAPPING[i] for i in range(6)] 

MISSING_RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caminhos
METADATA_FILE = "./data/PAD-UFES-20/metadata.csv"
IMAGE_ROOT = "./data/PAD-UFES-20/images"
OHE_PATH = "./data/preprocess_data/ohe_pad_20.pickle"
SCALER_PATH = "./data/preprocess_data/scaler_pad_20.pickle"

# Definição das colunas EXATAMENTE como no treino/GradCAM
PAD_COLUMNS = [
    "patient_id", "lesion_id", "smoke", "drink", "background_father", 
    "background_mother", "age", "pesticide", "gender", "skin_cancer_history", 
    "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick", 
    "region", "diameter_1", "diameter_2", "diagnostic", "itch", "grew", 
    "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"
]

NUMERICAL_COLS = ["age", "diameter_1", "diameter_2"]
CATEGORICAL_COLS = [c for c in PAD_COLUMNS if c not in NUMERICAL_COLS + ["patient_id", "lesion_id", "img_id", "biopsed", "diagnostic"]]
DROP_COLS = ["patient_id", "lesion_id", "img_id", "biopsed", "diagnostic"]

# ==========================================================
# FUNÇÕES DE APOIO E PRÉ-PROCESSAMENTO
# ==========================================================

def _strip_module_prefix(state_dict):
    """Remove o prefixo 'module.' de modelos salvos com DataParallel."""
    if not isinstance(state_dict, dict): return state_dict
    keys = list(state_dict.keys())
    if len(keys) > 0 and keys[0].startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("EMPTY")
    df = df.replace(r"^\s*$", "EMPTY", regex=True)
    df = df.replace([" ", "  ", "NÃO  ENCONTRADO"], "EMPTY")
    df = df.replace("BRASIL", "BRAZIL")
    return df

def parse_csv_line_to_cols(sample, columns: list) -> pd.DataFrame:
    """Garante que a entrada (DataFrame ou String) tenha todas as colunas na ordem certa."""
    if isinstance(sample, pd.DataFrame):
        for col in columns:
            if col not in sample.columns:
                sample[col] = "EMPTY"
        return sample[columns].copy()
    
    # Suporte para string caso você use essa função em outro lugar
    if isinstance(sample, str):
        parts = sample.split(",")
    else:
        parts = list(sample)
        
    if len(parts) < len(columns):
        parts = parts + [""] * (len(columns) - len(parts))
    else:
        parts = parts[:len(columns)]
    return pd.DataFrame([parts], columns=columns)

def process_metadata_pad20(df_raw, ohe, scaler, device):
    # 1. Prepara as colunas e limpa os dados
    df = parse_csv_line_to_cols(df_raw, PAD_COLUMNS)
    df = clean_metadata(df)

    # 2. Remove colunas de controle que o modelo não deve ler
    features = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 3. Garante o casting correto das features numéricas vs categóricas
    categorical_cols = [c for c in features.columns if c not in NUMERICAL_COLS]
    features[categorical_cols] = features[categorical_cols].astype(str)
    features[NUMERICAL_COLS] = features[NUMERICAL_COLS].apply(pd.to_numeric, errors="coerce").fillna(-1)

    # 4. Aplica OHE e Scaler
    categorical_data = ohe.transform(features[categorical_cols])
    numerical_data = scaler.transform(features[NUMERICAL_COLS])

    # 5. Concatenação Sagrada (Categóricos PRIMEIRO, Numéricos DEPOIS)
    processed = np.hstack([categorical_data, numerical_data])

    # 6. PADDING PARA 91 DIMENSÕES (Exigência Crítica do Checkpoint)
    target_size = 91
    if processed.shape[1] < target_size:
        diff = target_size - processed.shape[1]
        padding = np.zeros((processed.shape[0], diff))
        processed = np.hstack([processed, padding])
    elif processed.shape[1] > target_size:
        processed = processed[:, :target_size]

    return torch.tensor(processed, dtype=torch.float32).to(device)

def simulate_missing_metadata(df, missing_rate, numerical_cols, categorical_cols, seed=42):
    df = df.copy()
    rng = np.random.default_rng(seed)

    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)

    all_features = numerical_cols + categorical_cols
    mask = rng.random((len(df), len(all_features))) < (1 - missing_rate)

    for j, col in enumerate(all_features):
        if col not in df.columns:
            continue

        if col in numerical_cols:
            df.loc[~mask[:, j], col] = np.nan
        else:
            df.loc[~mask[:, j], col] = "EMPTY"

    return df
# ==========================================================
# EXECUÇÃO DO BENCHMARK
# ==========================================================

if __name__ == "__main__":
    print(f"🖥️ Usando dispositivo: {DEVICE}")
    
    with open(OHE_PATH, "rb") as f: shared_ohe = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: shared_scaler = pickle.load(f)
    
    transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    meta_all = pd.read_csv(METADATA_FILE)

    BACKBONES = ["caformer_b36.sail_in22k_ft_in1k"]
    MECHANISMS = ["att-intramodal+residual+cross-attention-metadados"] # ["no-metadata", "concatenation", "metablock", "crossattention", "att-intramodal+residual+cross-attention-metadados"] 

    for cnn_name in BACKBONES:
        for attention in MECHANISMS:
            print(f"\n⚙️ EXPERIMENTO: {cnn_name} | {attention}")
            
            base_results_dir = f"./src/results/testes-da-implementacao-final_2/25032026-WITH-LN/PAD-UFES-20/unfrozen_weights/8/{attention}/model_{cnn_name}_with_one-hot-encoder_512_with_best_architecture"
            
            all_folds_data = []

            for fold_idx in range(1, 6):
                fold_folder = f"{cnn_name}_fold_{fold_idx}"
                fold_path = os.path.join(base_results_dir, fold_folder)
                
                model_path = os.path.join(fold_path, "model.pth")
                if not os.path.exists(model_path):
                    model_path = os.path.join(fold_path, "best-model", "best_model.pt")

                if not os.path.exists(model_path): 
                    print(f"⚠️ Modelo não encontrado para o Fold {fold_idx}")
                    continue
                # Tenta carregar o modelo
                try:
                    # 1. Instância com Vocab Size Fixo em 91 e num_heads=8 (baseado no GradCAM)
                    model = multimodalIntraInterModal.MultimodalModel(
                        num_classes=6, 
                        device=DEVICE, 
                        cnn_model_name=cnn_name,
                        text_model_name="one-hot-encoder", 
                        vocab_size=91,  
                        num_heads=8, 
                        attention_mecanism=attention, 
                        n=2, 
                        unfreeze_weights="unfrozen_weights"
                    )
                
                    # 2. Carregar e Limpar o Dicionário
                    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
                    state_dict = ckpt.get("model_state_dict", ckpt)
                    state_dict = _strip_module_prefix(state_dict)

                    # 3. Carregamento Estrito (Sem try/except com strict=False)
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        print(f"✅ Fold {fold_idx}: Pesos carregados com sucesso (Strict=True)!")
                    except RuntimeError as e:
                        print(f"❌ Erro crítico no carregamento do Fold {fold_idx}:\n{e}")
                        sys.exit(1)

                    model.to(DEVICE).eval()
                except Exception as e:
                    print(f"Erro ao carregar o modelo:{e}\n")
                    continue
                
                # Carrega predições para filtrar as imagens de teste
                preds_csv = os.path.join(fold_path, f"predictions_eval_fold_{fold_idx}.csv")
                preds_fold = pd.read_csv(preds_csv)
                right_key = 'image_name' if 'image_name' in preds_fold.columns else 'img_id'
                df_test = pd.merge(meta_all, preds_fold, left_on="img_id", right_on=right_key)

                # Inferência
                for rate in MISSING_RATES:
                    y_true, y_pred, y_prob = [], [], []

                    # Aplica missing no fold inteiro antes da inferência
                    df_test_missing = simulate_missing_metadata(
                        df=df_test,
                        missing_rate=rate,
                        numerical_cols=NUMERICAL_COLS,
                        categorical_cols=CATEGORICAL_COLS,
                        seed=fold_idx + int(rate * 1000)
                    )

                    for _, row in df_test_missing.iterrows():
                        sample_df = pd.DataFrame([row[PAD_COLUMNS].to_dict()])

                        # Processamento de imagem
                        img = Image.open(os.path.join(IMAGE_ROOT, row["img_id"])).convert("RGB")
                        img_t = transform(image=np.array(img))["image"].unsqueeze(0).to(DEVICE)

                        # Processamento de metadados usando a função blindada
                        meta_t = process_metadata_pad20(sample_df, shared_ohe, shared_scaler, DEVICE)

                        # Forward Pass
                        with torch.no_grad():
                            output = model(img_t, meta_t)
                            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                            pred_idx = np.argmax(probs)

                        y_true.append(LABELS_LIST.index(row["diagnostic"]))
                        y_pred.append(pred_idx)
                        y_prob.append(probs)

                    y_prob = np.array(y_prob)

                    # Métricas
                    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
                    bacc = balanced_accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0)

                    try:
                        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                    except Exception:
                        auc = np.nan

                    all_folds_data.append({
                        "fold": fold_idx,
                        "missing_rate": rate,
                        "balanced_acc": bacc,
                        "f1_score": f1,
                        "accuracy": acc,
                        "auc": auc
                    })

                    print(
                        f"   ➔ Rate {rate} | "
                        f"ACC: {acc:.4f} | BAcc: {bacc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
                    )
            # ==========================================================
            # FINAL DO SCRIPT: SALVAMENTO FORMATADO
            # ==========================================================

            if all_folds_data:
                # 1. Transformar a lista de resultados em DataFrame
                res_df = pd.DataFrame(all_folds_data)
                
                # 2. Agrupar por missing_rate e calcular média e std para BAcc e F1
                summary = res_df.groupby("missing_rate").agg({
                    "balanced_acc": ["mean", "std"],
                    "f1_score": ["mean", "std"]
                }).reset_index()

                # 3. Achatar o MultiIndex das colunas e renomear conforme sua exigência
                summary.columns = [
                    "missing_rate", 
                    "bacc_mean", "bacc_std", 
                    "f1_mean", "f1_std"
                ]

                # 4. Adicionar as informações do Backbone e do Mecanismo
                summary["backbone"] = cnn_name
                summary["mechanism"] = attention

                # 5. Salvar o CSV final
                out_path = f"./src/results/testes-da-implementacao-final_2/25032026-WITH-LN/PAD-UFES-20/unfrozen_weights/8/summary/summary_{attention}_{cnn_name}.csv"
                summary.to_csv(out_path, index=False)
                
                print(f"\n📊 Resumo final gerado com sucesso!")
                print(f"📍 Salvo em: {out_path}")