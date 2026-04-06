#!/bin/bash
set -e

# ============================================================
# Ambiente
# ============================================================
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Carrega o .env
set -a
source ./conf/.env
set +a

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🚀 Iniciando simulação de dados faltantes com os modelos ..."
LOG_FILE="logs/simulate_missingmetadata_pad_20_artigo2_03042026_model_${TIMESTAMP}.log"

# ============================================================
# Execução
# ============================================================
nohup python3 -u ./src/scripts/benchmark/interpretability/inference_all_folds.py \
  > "$LOG_FILE" 2>&1 &

echo "✅ Processo iniciado em background (PID $!)"
