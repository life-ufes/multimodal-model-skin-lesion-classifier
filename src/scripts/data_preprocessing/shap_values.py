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
# 1) Carregar Dataset (usa o mesmo pipeline do multimodal)
# ======================================================
dataset = SkinLesionDataset(
    metadata_file="./data/PAD-UFES-20/metadata.csv",
    img_dir="./data/PAD-UFES-20/images",
    bert_model_name="one-hot-encoder",
    image_encoder="davit-v2",
    drop_nan=False,
    size=(224,224),
    is_train=False
)

meta = dataset.metadata.copy()

# ======================================================
# 2) Carregar probabilidades da validação (df)
# ======================================================
df = pd.read_csv("./src/results/pad_metadata_with_probs_fold_1.csv")

target_cols = ["prob_NEV","prob_BCC","prob_ACK","prob_SEK","prob_SCC","prob_MEL"]

# ======================================================
# 3) Merge correto via img_id
# ======================================================
merged = pd.merge(
    meta,
    df[["img_id"] + target_cols],
    on="img_id",
    how="inner"
)

print("Merged shape:", merged.shape)

# ======================================================
# 4) Pegar só os metadados que têm predições
# ======================================================
mask = meta["img_id"].isin(merged["img_id"])
X = dataset.features[mask]      # features já OHE + SCALER
y = merged[target_cols].values

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

# ======================================================
# 5) Treinar surrogate
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([("model", model)])
pipeline.fit(X_train, y_train)

print("\nR² surrogate:", pipeline.score(X_test, y_test))

# ======================================================
# 6) SHAP
# ======================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test, check_additivity=False)

print("SHAP calculado para todas as classes!")

# ======================================================
# 7) RECONSTRUIR NOMES DAS FEATURES — AGORA 100% CORRETO
# ======================================================
base_dir = "../data/preprocess_data"

with open(f"{base_dir}/ohe.pickle", "rb") as f:
    ohe = pickle.load(f)
with open(f"{base_dir}/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

categorical_cols = list(ohe.feature_names_in_)  # EXACT INPUT COLS
numeric_cols = list(scaler.feature_names_in_)   # ['age','diameter_1','diameter_2']

ohe_feature_names = []
for col_name, categories in zip(categorical_cols, ohe.categories_):
    for cat in categories:
        cat_str = str(cat).replace(" ", "_").replace("/", "_")
        ohe_feature_names.append(f"{col_name}_{cat_str}")

feature_names = ohe_feature_names + numeric_cols

print("\nReconstruído:", len(feature_names), "features")
print("Esperado:", X.shape[1])

if len(feature_names) != X.shape[1]:
    print("\n❌ ERRO: mismatch de features!")
    raise SystemExit()
else:
    print("✔ Feature names OK!")

# ======================================================
# Criar pasta para salvar os gráficos
# ======================================================
save_dir = "./src/results/shap_plots"
os.makedirs(save_dir, exist_ok=True)


# ======================================================
# 8) Summary plot SHAP (por classe) — e SALVAR EM PNG
# ======================================================
for class_idx, class_name in enumerate(target_cols):
    print(f"Gerando SHAP Summary Plot: {class_name}")
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values[class_idx],
        X_test,
        feature_names=feature_names,
        show=False
    )
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/summary_{class_name}.png", dpi=300)
    plt.close()
    print(f"OK — salvo em {save_dir}/summary_{class_name}.png")


# ======================================================
# 9) Bar plot SHAP (top 20 features) — e SALVAR EM PNG
# ======================================================
for class_idx, class_name in enumerate(target_cols):
    print(f"Gerando SHAP Bar Plot: {class_name}")
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values[class_idx],
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False
    )
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bar_{class_name}.png", dpi=300)
    plt.close()
    print(f"OK — salvo em {save_dir}/bar_{class_name}.png")


# ======================================================
# 10) Heatmap SHAP Global — e SALVAR EM PNG
# ======================================================
print("Gerando Heatmap SHAP Global...")

# calcula magnitude média por classe
mean_abs_shap = np.array([
    np.abs(shap_values[i]).mean(axis=0)
    for i in range(len(target_cols))
]).T  # transpose para ficar: features x classes

plt.figure(figsize=(14, len(feature_names) * 0.25))
sns.heatmap(
    mean_abs_shap,
    cmap="viridis",
    yticklabels=feature_names,
    xticklabels=target_cols
)

plt.title("SHAP Heatmap — Metadados vs Classes")
plt.tight_layout()
plt.savefig(f"{save_dir}/heatmap_global.png", dpi=300)
plt.close()

print(f"OK — heatmap salvo em {save_dir}/heatmap_global.png")