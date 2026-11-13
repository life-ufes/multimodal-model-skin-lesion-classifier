import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ------------------------------------------------------------
# STYLE
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

# Paleta categórica (Okabe-Ito estendida, amigável a daltônicos)
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#7A3E9D", "#8C564B", "#B2B200", "#44546A", "#E15759",
]
CM_CMAP = LinearSegmentedColormap.from_list("cm_blue", ["#FFFFFF", "#08306B"])


def _resolve_classes(df, classes):
    """Ordena os nomes das classes pelo label_idx (evita desalinhar com unique())."""
    if classes:
        return list(classes)
    if "label" not in df.columns:
        raise RuntimeError("CLASSES vazio e coluna 'label' ausente no CSV")
    m = df.drop_duplicates("label_idx").sort_values("label_idx")
    return m["label"].tolist()


def plot_confusion_matrix(y_true, y_pred, classes, path, normalize=True):
    n = len(classes)
    cm_raw = confusion_matrix(y_true, y_pred, labels=range(n))
    row_sum = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_raw, row_sum, out=np.zeros(cm_raw.shape, float), where=row_sum > 0)
    data = cm_norm if normalize else cm_raw
    vmax = 1.0 if normalize else cm_raw.max()

    size = max(7, 0.75 * n + 2.5)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(data, cmap=CM_CMAP, vmin=0, vmax=vmax)

    # grade branca entre células
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # rótulos: eixo y inclui o suporte (n) de cada classe
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor", fontsize=16)
    ax.set_yticklabels([f"{c}  (n={int(s)})" for c, s in zip(classes, row_sum.ravel())], fontsize=16)
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("True label", labelpad=10)

    # texto das células: % em destaque + contagem abaixo
    compact = n > 8  # muitas classes: só % nas células, para fonte maior
    fs = 15 if compact else max(12, 17 - n)
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if cm_raw[i, j] == 0 and n > 6:
                continue  # reduz poluição visual em matrizes grandes
            color = "white" if v > 0.5 * vmax else "#1a1a1a"
            weight = "bold" if i == j else "normal"
            if normalize:
                txt = f"{v * 100:.0f}" if compact else f"{v * 100:.1f}%\n({cm_raw[i, j]})"
            else:
                txt = f"{cm_raw[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", color=color,
                    fontsize=fs, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=14)

    ax.set_title("Confusion Matrix", pad=14)

    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_scores, classes, path):
    n = len(classes)
    y_bin = label_binarize(y_true, classes=range(n))

    curves = []
    for i, label in enumerate(classes):
        if y_bin[:, i].sum() == 0:  # classe ausente: ROC indefinida
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        curves.append((label, fpr, tpr, auc(fpr, tpr), i))

    # micro-average
    fpr_mi, tpr_mi, _ = roc_curve(y_bin.ravel(), y_scores.ravel())
    auc_mi = auc(fpr_mi, tpr_mi)

    # macro-average (interpolação em grade comum)
    grid = np.linspace(0, 1, 500)
    mean_tpr = np.mean([np.interp(grid, f, t) for _, f, t, _, _ in curves], axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    auc_ma = auc(grid, mean_tpr)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor("white")
    ax.grid(True, color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)

    # legenda ordenada por AUC
    for label, fpr, tpr, a, i in sorted(curves, key=lambda c: -c[3]):
        ax.plot(fpr, tpr, lw=2.6, alpha=0.95, color=PALETTE[i % len(PALETTE)],
                label=f"{label}  (AUC = {a:.3f})")

    ax.plot(fpr_mi, tpr_mi, lw=3.2, ls=":", color="#333333",
            label=f"Micro-average  (AUC = {auc_mi:.3f})")
    ax.plot(grid, mean_tpr, lw=3.6, ls="-", color="black",
            label=f"Macro-average  (AUC = {auc_ma:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color="#999999", label="Chance")

    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One-vs-Rest)", pad=12)
    ax.legend(loc="lower right", fontsize=13, frameon=True, framealpha=0.95,
              edgecolor="#DDDDDD", title="Class", title_fontsize=14)

    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def evaluate_from_csv_golden(CSV_PATTERN: str, OUTPUT_DIR: str, CLASSES: list, NORMALIZE_CM=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(CSV_PATTERN))
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {CSV_PATTERN}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    if not {"label_idx", "prediction_idx"}.issubset(df.columns):
        raise RuntimeError("CSV must contain label_idx and prediction_idx")

    CLASSES = _resolve_classes(df, CLASSES)
    prob_cols = [f"prob_{c}" for c in CLASSES]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing probability column(s): {missing}")

    y_true = df["label_idx"].values.astype(int)
    y_pred = df["prediction_idx"].values.astype(int)
    y_scores = df[prob_cols].values

    # Softmax defensivo (igual ao treino)
    if not np.allclose(y_scores.sum(axis=1), 1.0):
        e = np.exp(y_scores - y_scores.max(axis=1, keepdims=True))
        y_scores = e / e.sum(axis=1, keepdims=True)

    plot_confusion_matrix(y_true, y_pred, CLASSES,
                          os.path.join(OUTPUT_DIR, "confusion_matrix.png"), NORMALIZE_CM)
    plot_roc(y_true, y_scores, CLASSES, os.path.join(OUTPUT_DIR, "roc_curve.png"))

    print("✔ Evaluation completed")
    print(f"✔ Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    CLASSES = []
    BASE = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/artigo_1_GFCAM/07022026/PAD-UFES-20/multiclass/unfrozen_weights/8/only-metadata/model_resnet-50_with_one-hot-encoder_512_with_best_architecture/resnet-50_fold_2"
    CSV_PATTERN = f"{BASE}/predictions_eval_fold_2.csv"
    OUTPUT_DIR = BASE
    NORMALIZE_CM = True

    evaluate_from_csv_golden(CSV_PATTERN, OUTPUT_DIR, CLASSES, NORMALIZE_CM)