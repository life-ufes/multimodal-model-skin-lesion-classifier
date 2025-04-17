import os
import csv
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

def save_model_and_metrics(model, metrics, model_name, base_dir, fold_num, all_labels, all_predictions, targets, data_val="val"):
    """
    Salva o modelo, as métricas, a matriz de confusão e a curva ROC em uma pasta específica.

    Args:
        model: O modelo treinado (torch.nn.Module).
        metrics: Dicionário contendo as métricas de avaliação.
        model_name: Nome do modelo.
        base_dir: Diretório base onde as pastas serão criadas.
        fold_num: Número do fold (para validação cruzada).
        all_labels: Labels verdadeiros.
        all_predictions: Predições do modelo (probabilidades ou logits).
        targets: Lista de rótulos para exibição na matriz de confusão.
        data_val: Tipo de dados ('val' ou 'test').
    """
    # Criar diretório com timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(base_dir, f"{model_name}_fold{fold_num}_{data_val}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    # Salvar modelo
    model_file = os.path.join(save_path, "model.pth")
    torch.save(model.state_dict(), model_file)

    # Salvar métricas em CSV
    metrics_file = os.path.join(save_path, "metrics.csv")
    with open(metrics_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    # Matriz de confusão
    cm = confusion_matrix(all_labels, np.argmax(all_predictions, axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=targets)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format="d")
    plt.title("Matriz de Confusão")
    fig_cm.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=400)
    plt.close(fig_cm)

    # Curva ROC
    y_true = label_binarize(all_labels, classes=range(len(targets)))
    if all_predictions.shape[1] == 1:
        # Caso binário
        fpr, tpr, _ = roc_curve(y_true, all_predictions)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Curva ROC")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, "roc_curve.png"), dpi=400)
        plt.close()
    else:
        # Multiclasse
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(targets)):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], all_predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot
        fig_roc = plt.figure()
        for i in range(len(targets)):
            plt.plot(fpr[i], tpr[i], lw=2, label=f"{targets[i]} (area = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Curva ROC Multiclasse")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, "roc_curve.png"), dpi=400)
        plt.close(fig_roc)