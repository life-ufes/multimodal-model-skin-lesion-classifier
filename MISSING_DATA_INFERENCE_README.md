# Missing Data Inference - Multimodal Model

## Descrição

Este script realiza inferências com o modelo multimodal simulando diferentes níveis de dados faltantes em metadados (de 0% a 70%). Ele foi corrigido para funcionar corretamente com todos os 5 folds disponíveis.

## Correções Realizadas

### 1. **Assinatura da Função `process_metadata_pad20`**
   - **Problema**: A função foi definida para receber `text_line` (string) e `encoder_dir`, mas estava sendo chamada com um DataFrame e caminhos diretos aos encoders.
   - **Solução**: Refatorada para aceitar `df: pd.DataFrame`, `ohe_path: str`, `scaler_path: str` e `device`.

### 2. **Parâmetro de Atenção com Typo**
   - **Problema**: Existiam dois parâmetros conflitantes: `attention_mechanism` e `attention_mecanism` (com typo).
   - **Solução**: Unificados em um único parâmetro `attention_mechanism` em toda a função e suas chamadas.

### 3. **Duplicação de Definições de Variáveis**
   - **Problema**: `NUMERICAL_COLS` e `CATEGORICAL_METADATA_COLS` eram definidas duas vezes no arquivo.
   - **Solução**: Removidas as definições duplicadas.

### 4. **Configuração de Folds**
   - **Problema**: Original tinha apenas fold 3 hardcoded.
   - **Solução**: Agora detecta automaticamente e processa os 5 folds (pode ser customizado).

### 5. **Melhorias Adicionadas**
   - Carregamento automático de modelos disponíveis
   - Verificação de existência de arquivos antes de processar
   - Feedback visual do progresso com ✓ e ✗
   - Resumo de resultados agrupados por missing_rate
   - Tratamento de erros com try/except

## Como Usar

### Instalação de Dependências

```bash
cd /home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier
source venv/bin/activate
pip install -r requirements.txt
```

### Executar Inferências

```bash
cd /home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier
source venv/bin/activate
python3 src/scripts/benchmark/interpretability/inference_all_folds.py
```

### Customizar Folds a Processar

Edite o arquivo `src/scripts/benchmark/interpretability/inference_all_folds.py` e modifique:

```python
# Linha ~398
folds_to_process = [1, 2, 3, 4, 5]  # Customize para incluir apenas folds específicos
```

Por exemplo, para processar apenas folds 1 e 2:
```python
folds_to_process = [1, 2]
```

## Taxas de Dados Faltantes

O script testa as seguintes taxas de dados faltantes:
- **0.0** (0% - sem dados faltantes)
- **0.1** (10%)
- **0.2** (20%)
- **0.3** (30%)
- **0.5** (50%)
- **0.7** (70%)

Para modificar as taxas, edite:
```python
# Linha ~418
missing_rates = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
```

## Estrutura de Saída

O script salva os resultados em: `./src/results/missing_metadata_experiment.csv`

Colunas de saída:
- `fold`: Identificador do fold (1-5)
- `backbone`: Nome do backbone CNN (ex: caformer_b36.sail_in22k_ft_in1k)
- `attention_mechanism`: Tipo de mecanismo de atenção (ex: att-intramodal)
- `missing_rate`: Taxa de dados faltantes (0.0-0.7)
- `acc`: Acurácia
- `bacc`: Acurácia Balanceada
- `f1`: F1-Score (weighted)
- `auc`: AUC (macro)
- `model_size_mb`: Tamanho do modelo em MB

## Funcionamento

### 1. Simulação de Dados Faltantes
```python
# Função apply_missing_metadata
# Marca aleatoriamente uma fração de colunas como "faltando":
# - Colunas numéricas: preenchidas com -1
# - Colunas categóricas: preenchidas com "EMPTY"
```

### 2. Processamento de Metadados
```python
# Função process_metadata_pad20
# Realiza limpeza, one-hot encoding e normalização dos dados
```

### 3. Inferência do Modelo
```python
# Função inference
# Executa forward pass e retorna predições com probabilidades
```

## Notas Importantes

1. **GPU**: O script usa GPU se disponível (recomendado para faster inference)
2. **Tempo de Execução**: Cada fold com 6 missing rates pode levar vários minutos
3. **Reprodutibilidade**: Seeds são baseadas em (fold_id, img_id, missing_rate, idx) para garantir consistência
4. **Tratamento de Erros**: Se um fold falhar, o script continua com os próximos

## Troubleshooting

### Erro: "FileNotFoundError: OneHotEncoder or StandardScaler not found"
- Verifique os caminhos em `ohe_path` e `scaler_path`
- Certifique-se que os arquivos pickle existem em `./data/preprocess_data/`

### Erro: "FileNotFoundError: Model file not found"
- Verifique se o caminho do modelo está correto
- Confirme que o fold especificado exists nos resultados

### Aviso: "InconsistentVersionWarning: sklearn version mismatch"
- Isso é um aviso, não é crítico. Os modelos devem funcionar mesmo assim
- Para eliminar, reinstale sklearn com a mesma versão usada durante o treinamento

## Referências

- **Modelo**: Multimodal CNN-Transformer com atenção intra/inter-modal
- **Dataset**: PAD-UFES-20
- **Backbone CNN**: CAFormer B36
- **Encoder de Texto**: One-Hot Encoder
