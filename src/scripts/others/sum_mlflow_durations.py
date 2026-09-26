"""
Soma o tempo total de treino (duration) das runs do MLflow.
Requisitos: pip install mlflow pandas
Uso:
    python sum_mlflow_durations.py
"""
import mlflow
import pandas as pd

# Aponta para o seu servidor local
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient()

# --- 1) Liste os experimentos e ache os IDs dos 3 (Full, Last-10, Top-10) ---
experiments = client.search_experiments()
for exp in experiments:
    print(exp.experiment_id, "-", exp.name)

# --- 2) IDs dos 3 experimentos (Full, Last-10, Top-10), já mapeados por você ---
EXPERIMENT_DICT = {
    "938888365267774480": "LAST_K",  # LAST-10
    "833800452215545526": "TOP_K",   # TOP-10
    "364465972705935663": "FULL",    # FULL history
    "740398166983312696": "RANDOM_SEARCH",
}
EXPERIMENT_IDS = list(EXPERIMENT_DICT.keys())

# --- 3) Busca todas as runs (paginando automaticamente) ---
all_runs = mlflow.search_runs(
    experiment_ids=EXPERIMENT_IDS,
    filter_string="",
    max_results=50000,
)

# --- Duração: robusto para datetime (Timedelta) OU epoch em ms ---
delta = all_runs["end_time"] - all_runs["start_time"]
if pd.api.types.is_timedelta64_dtype(delta):
    all_runs["duration_min"] = delta.dt.total_seconds() / 60
else:
    all_runs["duration_min"] = delta / 1000 / 60

# Mapeia experiment_id -> nome legível do tipo de histórico
all_runs["history_type"] = all_runs["experiment_id"].map(EXPERIMENT_DICT)

# --- Diagnóstico de runs aninhadas (parent/child) ---
print("\nColunas disponíveis:")
print(all_runs.columns.tolist())
if "tags.mlflow.parentRunId" in all_runs.columns:
    n_child = all_runs["tags.mlflow.parentRunId"].notna().sum()
    print(f"\nRuns com parent (aninhadas): {n_child} de {len(all_runs)}")

total_minutes = float(all_runs["duration_min"].sum())
total_hours = total_minutes / 60
print(f"\nTotal de runs somadas: {len(all_runs)}")
print(f"Tempo total (todas as runs, todos os historicos): {total_minutes:.1f} min  ({total_hours:.2f} h)")

# --- 4) Soma por tipo de historico (Full / Last-10 / Top-10) ---
by_history = (
    all_runs.groupby("history_type")["duration_min"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "total_min", "count": "n_runs"})
)
by_history["total_h"] = by_history["total_min"] / 60
print("\nPor tipo de historico:")
print(by_history)

# --- 5) Soma por controlador LLM ---
if "params.controller_llm_name" in all_runs.columns:
    by_controller = (
        all_runs.groupby("params.controller_llm_name")["duration_min"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total_min", "count": "n_runs"})
    )
    by_controller["total_h"] = by_controller["total_min"] / 60
    print("\nPor controlador LLM (todos os historicos somados):")
    print(by_controller.sort_values("total_min", ascending=False))

    by_hist_controller = (
        all_runs.groupby(["history_type", "params.controller_llm_name"])["duration_min"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total_min", "count": "n_runs"})
    )
    by_hist_controller["total_h"] = by_hist_controller["total_min"] / 60
    print("\nPor historico x controlador LLM:")
    print(by_hist_controller.sort_values("total_min", ascending=False))
else:
    print("\n[AVISO] Coluna 'params.controller_llm_name' nao encontrada.")
    print("Veja a lista de colunas acima e me diga o nome real do parametro do controlador.")


# quantas child runs por run-pai, e nomes das runs
print(all_runs.groupby("history_type").size())
print(all_runs["tags.mlflow.runName"].str[:40].value_counts().head(20))
# parents distintos
print(all_runs["tags.mlflow.parentRunId"].nunique(), "parent runs distintos")
print([c for c in all_runs.columns if c.startswith("params.")])

CTRL = "params.controller_llm_model_name"

# 1) Ver quais controladores existem em CADA experimento somado
print(all_runs.groupby(["history_type", CTRL]).size())

# 2) Custo médio de treinar UM modelo (5 folds) — isso é o que importa
#    Filtra runs válidas com BACC medido, agrupa por sessão de busca
custo_por_fold = all_runs["duration_min"].median()
print(f"\nCusto mediano por fold: {custo_por_fold:.2f} min")
print(f"Custo estimado por modelo (5 folds): {custo_por_fold*5:.1f} min")

# 3) Argumento final: modelos treinados x custo
#    RS treinou 500 modelos; um controlador LLM bom, ~22
print(f"\nRandom Search (500 modelos): {500*custo_por_fold*5/60:.1f} h de treino")
print(f"Qwen3-0.6B (22 modelos): {22*custo_por_fold*5/60:.1f} h de treino + inferencia LLM")