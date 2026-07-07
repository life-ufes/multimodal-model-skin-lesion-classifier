#!/bin/bash
#
# Script para executar o treinamento de modelos em background
# 
# Uso: ./train_models_via_bash.sh [--wait] [--verbose]
# 
# Opções:
#   --wait      Aguarda a conclusão do processo (não rodará em background)
#   --verbose   Exibe logs em tempo real
#   --help      Mostra esta mensagem
#

set -o pipefail

# ============================================================
# CONFIGURAÇÕES
# ============================================================
# Encontra o diretório raiz do projeto (onde está o arquivo .env)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"

# Tenta git rev-parse como fallback
if [[ ! -f "$SCRIPT_DIR/conf/.env" ]] || [[ ! -f "$SCRIPT_DIR/logs" ]]; then
    if command -v git &> /dev/null; then
        SCRIPT_DIR="$(cd "$(git rev-parse --show-toplevel 2>/dev/null)" && pwd)"
    fi
fi

cd "$SCRIPT_DIR" || exit 1

SCRIPT_FILE="${BASH_SOURCE[0]}"
PYTHON_SCRIPT="./src/scripts/benchmark/train_pad_20.py"
PYTHON_SCRIPT_ABSOLUTE="$SCRIPT_DIR/src/scripts/benchmark/train_pad_20.py"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_pad_20_${TIMESTAMP}.log"

# Flags
RUN_IN_BACKGROUND=true
VERBOSE=false

# ============================================================
# CORES PARA OUTPUT
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# FUNÇÕES
# ============================================================

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

show_help() {
    cat << 'EOF'
Script para executar o treinamento de modelos em background

Uso: ./train_models_via_bash.sh [--wait] [--verbose]

Opções:
  --wait      Aguarda a conclusão do processo (não rodará em background)
  --verbose   Exibe logs em tempo real
  --help      Mostra esta mensagem
EOF
}

cleanup_old_logs() {
    local keep_days=7
    print_info "Limpando logs com mais de ${keep_days} dias..."
    find "${LOG_DIR}" -name "train_*.log" -type f -mtime +"${keep_days}" -delete 2>/dev/null || true
}

# ============================================================
# PARSE ARGUMENTOS
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --wait)
            RUN_IN_BACKGROUND=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Argumento desconhecido: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================
# VALIDAÇÕES PRÉ-REQUISITOS
# ============================================================
print_header "Validando Pré-Requisitos"

# Debug: mostra diretório de trabalho
print_info "Diretório de trabalho: $(pwd)"
print_info "Procurando script em: $PYTHON_SCRIPT"

# Verifica Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 não está instalado ou não está no PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
print_success "Python3 encontrado: $PYTHON_VERSION"

# Verifica se está em um virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    print_warning "⚠️  Nenhum virtual environment ativo"
    print_info "Se PyTorch não está instalado globalmente, ative o venv:"
    echo ""
    echo "    source venv/bin/activate"
    echo ""
fi

# Verifica se script Python existe (tenta múltiplos caminhos)
PYTHON_SCRIPT_FINAL=""
if [[ -f "$PYTHON_SCRIPT" ]]; then
    PYTHON_SCRIPT_FINAL="$PYTHON_SCRIPT"
elif [[ -f "$PYTHON_SCRIPT_ABSOLUTE" ]]; then
    PYTHON_SCRIPT_FINAL="$PYTHON_SCRIPT_ABSOLUTE"
else
    print_error "Script Python não encontrado em nenhum dos caminhos:"
    print_error "  Relativo: $PYTHON_SCRIPT"
    print_error "  Absoluto: $PYTHON_SCRIPT_ABSOLUTE"
    print_error ""
    print_info "Procurando por arquivos train_pad_*.py no projeto..."
    find "$SCRIPT_DIR" -maxdepth 3 -name "train_pad_*.py" -type f 2>/dev/null | head -5 | sed 's/^/  /'
    exit 1
fi

PYTHON_SCRIPT="$PYTHON_SCRIPT_FINAL"
print_success "Script Python encontrado: $PYTHON_SCRIPT"

# Verifica se torch está disponível
if ! python3 -c "import torch" 2>/dev/null; then
    print_error "PyTorch não está instalado"
    print_info "Para instalar, execute:"
    echo "    pip install -r requirements.txt"
    print_info "Ou, se estiver em um virtual environment:"
    echo "    source venv/bin/activate"
    echo "    pip install -r requirements.txt"
    exit 1
fi
print_success "PyTorch encontrado"

# Verifica diretório de logs
if [[ ! -d "$LOG_DIR" ]]; then
    print_info "Criando diretório de logs: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi
print_success "Diretório de logs: $LOG_DIR"

# Carrega .env
if [[ -f "./conf/.env" ]]; then
    set -a
    source ./conf/.env
    set +a
    print_success "Arquivo .env carregado"
else
    print_warning "Arquivo .env não encontrado em ./conf/.env"
fi

# ============================================================
# CONFIGURAÇÃO DO AMBIENTE
# ============================================================
print_header "Configurando Ambiente"

export PYTHONUNBUFFERED=1
print_success "PYTHONUNBUFFERED=1"

export CUDA_VISIBLE_DEVICES=0
print_success "CUDA_VISIBLE_DEVICES=0"

# Verifica GPU (opcional)
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detectada"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^/  /'
else
    print_warning "nvidia-smi não encontrado (CPU será usado)"
fi

# ============================================================
# EXECUÇÃO
# ============================================================
print_header "Iniciando Treinamento"

echo "Parâmetros:"
echo "  Script: $PYTHON_SCRIPT"
echo "  Log: $LOG_FILE"
echo "  Background: $RUN_IN_BACKGROUND"
echo "  Verbose: $VERBOSE"
echo ""

# Cleanup logs antigos
cleanup_old_logs

# Captura o processo ID e controla melhor
if [[ "$RUN_IN_BACKGROUND" == true ]]; then
    # Executa em background
    python3 "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &
    PID=$!
    
    print_success "Processo iniciado com PID: $PID"
    print_info "Logs sendo salvos em: $LOG_FILE"
    echo ""
    
    # Aguarda um pouco para verificar se process não falhou no início
    sleep 2
    if ! kill -0 $PID 2>/dev/null; then
        print_error "Processo falhou ao iniciar. Verificando logs..."
        tail -20 "$LOG_FILE"
        exit 1
    fi
    
    print_success "Processo rodando normalmente"
    echo ""
    
    if [[ "$VERBOSE" == true ]]; then
        print_info "Monitorando logs em tempo real (Ctrl+C para sair)..."
        tail -f "$LOG_FILE"
    else
        print_info "Para monitorar o progresso, execute:"
        echo "  tail -f $LOG_FILE"
    fi
else
    # Executa em foreground e aguarda conclusão
    print_info "Aguardando conclusão do processo..."
    echo ""
    
    if python3 -u "$PYTHON_SCRIPT" 2>&1 | tee "$LOG_FILE"; then
        print_success "Treinamento concluído com sucesso!"
        EXIT_CODE=0
    else
        print_error "Treinamento falhou!"
        EXIT_CODE=$?
    fi
    
    exit $EXIT_CODE
fi

# ============================================================
# FIM
# ============================================================
echo ""
print_success "Script de treinamento configurado e em execução"
