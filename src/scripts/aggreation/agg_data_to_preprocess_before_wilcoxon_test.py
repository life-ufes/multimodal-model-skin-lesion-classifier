"""
Friedman + Wilcoxon (GFCAM vs. métodos da Tabela de comparação, PAD-UFES-20).

As amostras de cada método são os 5 folds de cada métrica, concatenados na mesma ordem
(ACC dos 5 folds, depois BACC, depois AUC). Assim, a comparação é pareada por fold e métrica.
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd

import stats  # módulo local

# ROOT = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/artigo_1_GFCAM/07022026/PAD-UFES-20/unfrozen_weights/8"
ROOT = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/artigo_1_GFCAM/07022026/PAD-UFES-20/multiclass/unfrozen_weights/8"
# OUT_DIR = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/tests/results_08112026"
OUT_DIR = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/artigo_1_GFCAM/07022026/PAD-UFES-20/multiclass/"
METRICS = ("accuracy", "balanced_accuracy", "f1_score", "auc")   # as mesmas do artigo e do A-TOPSIS
N_FOLDS = 5


METHODS = {
   "GFCAM (ours)":    ("gfcam",          "densenet169"),
   "No metadata":     ("no-metadata",    "resnet-50"),
   "No image":        ("only-metadata",  "resnet-50"),
   "Concatenation":   ("concatenation",  "resnet-50"),
   "MetaBlock":       ("metablock",      "efficientnet-b4"),
   "MD-Net":          ("md-net",         "densenet169"),
   "Cross-Attention": ("crossattention", "resnet-50"),
   "LiwTERM":         ("liwterm",        "resnet-50"),
}

# METHODS = {
#     "GFCAM (ours)":    ("gfcam",          "resnet-50"),
#     "No metadata":     ("no-metadata",    "resnet-50"),
#     "No image":        ("only-metadata",  "resnet-50"),
#     "Concatenation":   ("concatenation",  "resnet-50"),
#     "MetaBlock":       ("metablock",      "resnet-50"),
#     "MD-Net":          ("md-net",         "resnet-50"),
#     "Cross-Attention": ("crossattention", "resnet-50"),
#     "LiwTERM":         ("liwterm",        "resnet-50"),
# }

REFERENCE = "GFCAM (ours)"


def model_dir(folder, backbone):
    return os.path.join(ROOT, folder, f"model_{backbone}_with_one-hot-encoder_512_with_best_architecture")


def load_folds(path, metrics=METRICS, file_name="model_metrics.csv"):
    """Retorna (vetor concatenado por métrica, DataFrame dos folds)."""
    csv_file = os.path.join(path, file_name)
    if not os.path.exists(csv_file):
        raise FileNotFoundError(csv_file)
    df = pd.read_csv(csv_file)
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise KeyError(f"Métricas ausentes em {csv_file}: {missing}")
    if len(df) != N_FOLDS:
        raise ValueError(f"{csv_file}: esperados {N_FOLDS} folds, encontrados {len(df)}")
    # Se houver coluna de fold, ordena por ela para garantir o pareamento
    for col in ("fold", "fold_id", "k"):
        if col in df.columns:
            df = df.sort_values(col)
            break
    return np.concatenate([df[m].to_numpy(float) for m in metrics]), df


def main():
    names, data, summary = [], [], []
    for name, (folder, backbone) in METHODS.items():
        vec, df = load_folds(model_dir(folder, backbone))
        names.append(name)
        data.append(vec)
        summary.append({"method": name, "backbone": backbone,
                        **{m: f"{df[m].mean():.4f} ± {df[m].std(ddof=0):.4f}" for m in METRICS}})

    # Confira estes valores com a tabela do artigo antes de usar os p-valores
    print(pd.DataFrame(summary).to_string(index=False), "\n")

    p_friedman, res = stats.statistical_test(data, names, reference=REFERENCE, correction="holm")

    os.makedirs(OUT_DIR, exist_ok=True)
    res.insert(0, "friedman_p", p_friedman)
    out_path = os.path.join(OUT_DIR, f"gfcam_vs_literature_wilcoxon_11112026.csv")
    res.to_csv(out_path, index=False)
    print(f"\nResultados salvos em {out_path}")


if __name__ == "__main__":
    main()