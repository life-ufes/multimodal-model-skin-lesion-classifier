import os
import csv
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

def save_model_and_metrics(model,
                           metrics:dict,
                           model_name:str,
                           save_to_disk:bool,
                           base_dir:str,
                           fold_num:int,
                           all_labels,
                           all_predictions,
                           targets: list,
                           data_val:str="val"):
    """
    Salva modelo, métricas, matriz de confusão e curva ROC em 400 dpi,
    tratando corretamente shapes 1-D, (n,1), (n,2) e multiclasses.
    """
    # --- Preparação
    all_labels      = np.array(all_labels)
    all_predictions = np.array(all_predictions)

        # Gerar o nome único da pasta usando o nome do modelo e o timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_name}_fold_{fold_num}_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)

    # Criar a pasta para o modelo
    os.makedirs(folder_path, exist_ok=True)

    if save_to_disk is True:
        # Salvar o modelo treinado
        model_path = os.path.join(folder_path, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em: {model_path}")

    # Salvar as métricas
    metrics_file = os.path.join(base_dir, "model_metrics.csv")
    file_exists = os.path.isfile(metrics_file)
    with open(metrics_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Métricas salvas em: {metrics_file}")

    # --- Predições discretas para Confusion Matrix ---
    # 1-D ou 1-coluna: threshold 0.5 binário
    if all_predictions.ndim == 1 or all_predictions.shape[1] == 1:
        preds = (all_predictions.flatten() > 0.5).astype(int)
    else:
        # qualquer outra shape (n,≥2): argmax
        preds = np.argmax(all_predictions, axis=1)

    # Matriz de Confusão
    cm   = confusion_matrix(all_labels, preds, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=targets)
    fig, ax = plt.subplots(figsize=(8,6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".4f")
    ax.set_title("Confusion Matrix")
    fig.savefig(os.path.join(folder_path, "confusion_matrix.png"), dpi=400)
    plt.close(fig)

    # --- Curva ROC ---
    # Define se é binário puro, binário com duas colunas ou multiclasses
    if all_predictions.ndim == 1 or all_predictions.shape[1] == 1:
        # Caso A: 1-D ou (n,1)
        y_true   = all_labels
        y_scores = all_predictions.flatten()

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc     = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0,1],[0,1], lw=2, linestyle="--", label="Random")
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(folder_path, "roc_curve.png"), dpi=400)
        plt.close(fig)

    elif all_predictions.shape[1] == 2:
        # Caso B: (n,2) → pega coluna 1 como score
        y_true   = all_labels
        y_scores = all_predictions[:,1]

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc     = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0,1],[0,1], lw=2, linestyle="--", label="Random")
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(folder_path, "roc_curve.png"), dpi=400)
        plt.close(fig)

    else:
        # Caso C: multiclasses (>2 colunas)
        n_classes = all_predictions.shape[1]
        y_true_bin = label_binarize(all_labels, classes=range(n_classes))

        # Garantir probabilidades
        if not np.allclose(all_predictions.sum(axis=1), 1.0):
            y_scores = torch.softmax(torch.tensor(all_predictions), dim=1).numpy()
        else:
            y_scores = all_predictions

        fpr = {}; tpr = {}; roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_scores[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, ax = plt.subplots(figsize=(8,6))
        for i, label in enumerate(targets):
            ax.plot(fpr[i], tpr[i], lw=2, label=f"{label} (AUC={roc_auc[i]:.2f})")
        ax.plot([0,1],[0,1], lw=2, linestyle="--", label="Random")
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(folder_path, "roc_curve.png"), dpi=400)
        plt.close(fig)

    return folder_path