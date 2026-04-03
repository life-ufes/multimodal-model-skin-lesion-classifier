#!/usr/bin/env python3
"""
Script de validação para o sistema de inferência com dados faltantes.
Verifica se todos os componentes estão funcionando corretamente.
"""
import os
import sys
import pickle

# Check Python version
print("=" * 60)
print("VALIDAÇÃO DO SISTEMA DE INFERÊNCIA COM DADOS FALTANTES")
print("=" * 60)

print(f"\nPython versão: {sys.version}")

# Check imports
print("\n1. Verificando imports...")
try:
    import numpy as np
    print("   ✓ NumPy")
except ImportError as e:
    print(f"   ✗ NumPy: {e}")

try:
    import pandas as pd
    print("   ✓ Pandas")
except ImportError as e:
    print(f"   ✗ Pandas: {e}")

try:
    import torch
    print(f"   ✓ PyTorch (versão {torch.__version__})")
    print(f"     CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     CUDA devices: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"   ✗ PyTorch: {e}")

try:
    from PIL import Image
    print("   ✓ PIL")
except ImportError as e:
    print(f"   ✗ PIL: {e}")

try:
    import albumentations
    print("   ✓ Albumentations")
except ImportError as e:
    print(f"   ✗ Albumentations: {e}")

try:
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    print("   ✓ scikit-learn")
except ImportError as e:
    print(f"   ✗ scikit-learn: {e}")

# Check required files and paths
print("\n2. Verificando arquivos necessários...")

base_paths = {
    "Metadata": "./data/PAD-UFES-20/metadata.csv",
    "Images": "./data/PAD-UFES-20/images",
    "OneHotEncoder": "./data/preprocess_data/ohe_pad_20.pickle",
    "StandardScaler": "./data/preprocess_data/scaler_pad_20.pickle",
}

for name, path in base_paths.items():
    exists = os.path.exists(path)
    symbol = "✓" if exists else "✗"
    print(f"   {symbol} {name}: {path}")

# Check model paths
print("\n3. Verificando modelos disponíveis...")

base_model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/25032026-WITH-LN/PAD-UFES-20/unfrozen_weights/8/att-intramodal"
available_folds = []

for fold in range(1, 6):
    fold_path = os.path.join(base_model_path, f"model_caformer_b36.sail_in22k_ft_in1k_with_one-hot-encoder_512_with_best_architecture/caformer_b36.sail_in22k_ft_in1k_fold_{fold}")
    model_file = os.path.join(fold_path, "best-model/best_model.pt")
    pred_file = os.path.join(fold_path, f"predictions_eval_fold_{fold}.csv")
    
    model_exists = os.path.exists(model_file)
    pred_exists = os.path.exists(pred_file)
    
    if model_exists and pred_exists:
        available_folds.append(fold)
        print(f"   ✓ Fold {fold}: Modelo e predições OK")
    elif model_exists or pred_exists:
        print(f"   ⚠ Fold {fold}: Apenas um arquivo disponível")
    else:
        print(f"   ✗ Fold {fold}: Arquivos não encontrados")

# Check encoders
print("\n4. Carregando encoders...")
try:
    with open("./data/preprocess_data/ohe_pad_20.pickle", "rb") as f:
        ohe = pickle.load(f)
    print(f"   ✓ OneHotEncoder carregado")
    print(f"     Features: {ohe.get_feature_names_out()[:5]}... (mostrando primeiras 5)")
except Exception as e:
    print(f"   ✗ OneHotEncoder: {e}")

try:
    with open("./data/preprocess_data/scaler_pad_20.pickle", "rb") as f:
        scaler = pickle.load(f)
    print(f"   ✓ StandardScaler carregado")
except Exception as e:
    print(f"   ✗ StandardScaler: {e}")

# Summary
print("\n" + "=" * 60)
print("RESUMO DA VALIDAÇÃO")
print("=" * 60)

if available_folds:
    print(f"\n✓ Folds disponíveis: {available_folds}")
    print(f"✓ Total de folds: {len(available_folds)}")
    print(f"\nO sistema está pronto para executar infer inferências!")
    print(f"Total de inferences: {len(available_folds)} folds × 6 missing rates = {len(available_folds) * 6}")
else:
    print("\n✗ Nenhum fold disponível!")
    print("Verifique os caminhos dos modelos e predições.")

print("\n" + "=" * 60)
