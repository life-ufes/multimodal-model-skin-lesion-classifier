import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from models.skinLesionDatasets import SkinLesionDataset

# ======================================================
# 1) Carregar Dataset
# ======================================================
dataset = SkinLesionDataset(
    metadata_file="./data/PAD-UFES-20/metadata.csv",
    img_dir="./data/PAD-UFES-20/images",
    bert_model_name="one-hot-encoder",
    image_encoder="caformer-b36",
    drop_nan=False,
    size=(224, 224),
    is_train=False
)

meta = dataset.metadata.copy()

# ======================================================
# 2) Carregar probabilidades da validação
# ======================================================
df = pd.read_csv(
    "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/"
    "src/results/testes-da-implementacao-final_2/01012026/"
    "PAD-UFES-20/unfrozen_weights/8/"
    "att-intramodal+residual+cross-attention-metadados/"
    "model_caformer_b36.sail_in22k_ft_in1k_with_one-hot-encoder_512_with_best_architecture/"
    "caformer_b36.sail_in22k_ft_in1k_fold_3/"
    "predictions_eval_fold_3.csv"
)

target_cols = [
    "prob_NEV", "prob_BCC", "prob_ACK",
    "prob_SEK", "prob_SCC", "prob_MEL"
]

# ======================================================
# 3) Merge correto via img_id
# ======================================================
target_id = "img_id"

merged = pd.merge(
    meta,
    df[[target_id] + target_cols],
    on=target_id,
    how="inner"
)

print("Merged shape:", merged.shape)

# ======================================================
# 4) Extrair X e y
# ======================================================
mask = meta[target_id].isin(merged[target_id])

X = dataset.features[mask]          # (N, 91)
y = merged[target_cols].values      # (N, 6)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

# ======================================================
# 5) Treinar surrogate
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

surrogate = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("model", surrogate)
])

pipeline.fit(X_train, y_train)

print("\nR² surrogate:", pipeline.score(X_test, y_test))

# ======================================================
# 6) SHAP — API moderna
# ======================================================
explainer = shap.Explainer(
    pipeline.predict,
    X_train
)

shap_exp = explainer(X_test)

print("SHAP calculado!")
print("SHAP values shape:", shap_exp.values.shape)

# ======================================================
# 7) Reconstruir nomes das features
# ======================================================
base_dir = "./data/preprocess_data"

with open(f"{base_dir}/ohe_pad_20.pickle", "rb") as f:
    ohe = pickle.load(f)

with open(f"{base_dir}/scaler_pad_20.pickle", "rb") as f:
    scaler = pickle.load(f)

categorical_cols = list(ohe.feature_names_in_)
numeric_cols = list(scaler.feature_names_in_)

ohe_feature_names = []
for col, cats in zip(categorical_cols, ohe.categories_):
    for cat in cats:
        cat_str = str(cat).replace(" ", "_").replace("/", "_")
        ohe_feature_names.append(f"{col}_{cat_str}")

feature_names = ohe_feature_names + numeric_cols

print("\nReconstruído:", len(feature_names), "features")
print("Esperado:", X.shape[1])

assert len(feature_names) == X.shape[1]
print("✔ Feature names OK!")

# ======================================================
# 8) Pasta de saída
# ======================================================
save_dir = (
    "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/"
    "src/results/shap_plots/rg-dermnet"
)
os.makedirs(save_dir, exist_ok=True)

# ======================================================
# 9) SHAP Summary Plot — PADRÃO CONSISTENTE
# ======================================================
for class_idx, class_name in enumerate(target_cols):
    print(f"Gerando SHAP Summary Plot: {class_name}")

    fig = plt.figure(figsize=(8, 10))

    shap.summary_plot(
        shap_exp.values[:, :, class_idx],
        X_test,
        feature_names=feature_names,
        show=False
    )

    fig.text(
        0.5, 0.96,
        f"SHAP Summary — {class_name.replace('prob_', '')}",
        ha="center",
        va="top",
        fontsize=14,
        weight="bold"
    )

    plt.subplots_adjust(top=0.90)
    plt.savefig(f"{save_dir}/summary_{class_name}.png", dpi=400)
    plt.close()


# ======================================================
# 10) SHAP Bar Plot — FIGSIZE + TÍTULO DEFINITIVO
# ======================================================
for class_idx, class_name in enumerate(target_cols):
    print(f"Gerando SHAP Bar Plot: {class_name}")

    fig = plt.figure(figsize=(7, 8))  # figura explícita

    shap.summary_plot(
        shap_exp.values[:, :, class_idx],
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False
    )

    # 🔥 TÍTULO FORA DO EIXO (ÚNICA FORMA QUE FUNCIONA)
    fig.text(
        0.5, 0.96,
        f"Mean |SHAP| — {class_name.replace('prob_', '')}",
        ha="center",
        va="top",
        fontsize=14,
        weight="bold"
    )

    plt.subplots_adjust(top=0.90)
    plt.savefig(f"{save_dir}/bar_{class_name}.png", dpi=400)
    plt.close()

# ======================================================
# 11) Heatmap SHAP Global (inalterado)
# ======================================================
print("Gerando Heatmap SHAP Global...")

mean_abs_shap = np.abs(shap_exp.values).mean(axis=0)  # (features, classes)

plt.figure(figsize=(14, len(feature_names) * 0.25))
sns.heatmap(
    mean_abs_shap,
    cmap="viridis",
    yticklabels=feature_names,
    xticklabels=target_cols
)

plt.title("SHAP Heatmap — Metadados vs Classes (Surrogate)")
plt.tight_layout()
plt.savefig(f"{save_dir}/heatmap_global.png", dpi=400)
plt.close()

print("✔ Todos os plots SHAP foram gerados com sucesso!")

# Importância das featuress globais
global_importance = np.abs(shap_exp.values).mean(axis=(0, 2))

df_global = pd.DataFrame({
    "feature": feature_names,
    "global_mean_abs_shap": global_importance
}).sort_values("global_mean_abs_shap", ascending=False)

print("\nTop 10 features globais")
print(df_global.head(10))


## Importância por classe
feature_importance_per_class = {}

for class_idx, class_name in enumerate(target_cols):
    mean_abs = np.abs(shap_exp.values[:, :, class_idx]).mean(axis=0)

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    feature_importance_per_class[class_name] = df_imp

    print(f"\nTop 10 features — {class_name}")
    print(df_imp.head(10))
