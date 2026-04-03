#!/bin/bash
# Script de conveniência para executar as inferências com dados faltantes
# Este script configura o ambiente e executa o pipeline completo

set -e  # Exit on error

cd "$(dirname "$0")"

echo "=========================================="
echo "FRAMEWORK DE INFERÊNCIA - DADOS FALTANTES"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Activate virtual environment
echo -e "${BLUE}[1/4] Ativando ambiente virtual...${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Ambiente virtual ativado${NC}"
else
    echo -e "${RED}✗ Ambiente virtual não encontrado!${NC}"
    echo "Criando novo ambiente virtual com Python 3.12..."
    python3.12 -m venv venv
    source venv/bin/activate
    echo "Instalando dependências..."
    pip install --upgrade pip
    pip install -r requirements.txt pandas numpy scikit-learn
fi
echo ""

# Step 2: Validate setup
echo -e "${BLUE}[2/4] Validando configuração do sistema...${NC}"
python3 validate_inference_setup.py
echo ""

# Step 3: Run inference
echo -e "${BLUE}[3/4] Executando inferências com dados faltantes...${NC}"
echo "Isso pode levar vários minutos (espere~30-75 minutos para 5 folds)."
echo ""

python3 src/scripts/benchmark/interpretability/inference_all_folds.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Inferências completadas com sucesso!${NC}"
else
    echo ""
    echo -e "${RED}✗ Erro durante as inferências${NC}"
    exit 1
fi

# Step 4: Show results
echo ""
echo -e "${BLUE}[4/4] Exibindo resultado resumido...${NC}"

if [ -f "src/results/missing_metadata_experiment.csv" ]; then
    echo ""
    echo "Resultados salvos em: src/results/missing_metadata_experiment.csv"
    echo ""
    echo "Resumo estatístico por taxa de dados faltantes:"
    echo "============================================================"
    
    # Use Python para exibir um resumo dos resultados
    python3 << 'EOF'
import pandas as pd
import os

csv_path = "src/results/missing_metadata_experiment.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    # Summary by missing rate
    summary = df.groupby('missing_rate')[['acc', 'bacc', 'f1', 'auc']].agg(['mean', 'std'])
    
    print("\nMédia ± Desvio Padrão por Taxa de Dados Faltantes:\n")
    print(f"{'Missing Rate':<15} {'Accuracy':<18} {'BACC':<18} {'F1-Score':<18} {'AUC':<18}")
    print("-" * 70)
    
    for missing_rate in sorted(df['missing_rate'].unique()):
        rate_data = df[df['missing_rate'] == missing_rate]
        
        acc_mean = rate_data['acc'].mean()
        acc_std = rate_data['acc'].std()
        bacc_mean = rate_data['bacc'].mean()
        bacc_std = rate_data['bacc'].std()
        f1_mean = rate_data['f1'].mean()
        f1_std = rate_data['f1'].std()
        auc_mean = rate_data['auc'].mean()
        auc_std = rate_data['auc'].std()
        
        missing_pct = f"{missing_rate*100:>5.0f}%"
        acc_str = f"{acc_mean:.4f}±{acc_std:.4f}"
        bacc_str = f"{bacc_mean:.4f}±{bacc_std:.4f}"
        f1_str = f"{f1_mean:.4f}±{f1_std:.4f}"
        auc_str = f"{auc_mean:.4f}±{auc_std:.4f}"
        
        print(f"{missing_pct:<15} {acc_str:<18} {bacc_str:<18} {f1_str:<18} {auc_str:<18}")
    
    print(f"\nTotal de resultados obtidos: {len(df)}")
    print(f"Folds processados: {sorted(df['fold'].unique())}")
else:
    print(f"Arquivo de resultados não encontrado: {csv_path}")
EOF
else
    echo -e "${RED}✗ Arquivo de resultados não encontrado${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✓ PIPELINE CONCLUÍDO COM SUCESSO!${NC}"
echo "=========================================="
echo ""
echo "Próximas etapas:"
echo "  1. Revisar resultados em: src/results/missing_metadata_experiment.csv"
echo "  2. Análisar a degradação de performance"
echo "  3. Comparar com baseline (missing_rate = 0.0)"
echo ""
