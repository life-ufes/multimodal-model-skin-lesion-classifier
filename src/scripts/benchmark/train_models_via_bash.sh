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

echo "🚀 Iniciando treino dos modelos otimizados ..."
LOG_FILE="logs/train_pad_20_artigo2_02042026_model_${TIMESTAMP}.log"

# ============================================================
# Execução
# ============================================================
nohup python3 -u ./src/scripts/benchmark/train_pad_20.py \
  > "$LOG_FILE" 2>&1 &

echo "✅ Processo iniciado em background (PID $!)"
