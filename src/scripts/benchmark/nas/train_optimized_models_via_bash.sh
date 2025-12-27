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
LOG_FILE="logs/train_pad_20_optimized_model_${TIMESTAMP}.log"

# ============================================================
# Execução
# ============================================================
nohup python3 -u ./src/scripts/benchmark/nas/train_pad_20_optimized_model.py \
  > "$LOG_FILE" 2>&1 &

echo "✅ Processo iniciado em background (PID $!)"
