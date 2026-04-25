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

# ============================================================
# Logs
# ============================================================
LOG_DIR=logs
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Sanitiza nome do modelo para nome de arquivo
SAFE_LLM_NAME=$(echo "$LLM_MODEL_NAME_SEQUENCE_GENERATOR" | sed 's/[:\/]/_/g')
HISTORY_MODE="last_k" # "full" # "top_k"

LOG_FILE="$LOG_DIR/nas_20042026_${TIMESTAMP}_${HISTORY_MODE}.log"

echo "🧠 LLM: $LLM_MODEL_NAME_SEQUENCE_GENERATOR"
echo "📄 Log: $LOG_FILE"
echo "🚀 Iniciando NAS..."

# ============================================================
# Execução
# ============================================================
nohup python3 -u ./src/scripts/benchmark/nas/train_pad_20_optimized_model.py \
  > "$LOG_FILE" 2>&1 &

echo "✅ Processo iniciado em background (PID $!)"
