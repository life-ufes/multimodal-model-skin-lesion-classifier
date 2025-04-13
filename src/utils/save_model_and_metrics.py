import os
import csv
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

def ensure_numpy(arr, name="array"):
    """Converte a entrada em um array NumPy se for uma lista."""
    if isinstance(arr, list):
        arr = np.array(arr)
        print(f"Conversão de {name} de lista para array NumPy realizada.")
    return arr

def create_folder(base_dir, model_name, fold_num):
    """Cria e retorna o caminho da pasta para salvar os resultados com um timestamp único."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_name}_fold_{fold_num}_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_csv(metrics, csv_file):
    """Salva/Anexa as métricas em um arquivo CSV."""
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Métricas salvas em: {csv_file}")

def plot_and_save_figure(fig, save_path):
    """Salva a figura e fecha-a para liberar memória."""
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Figura salva em: {save_path}")

def plot_confusion_matrix(all_labels, all_predictions, targets, folder_path, data_val="val", test_folder=None):
    """Calcula e salva a matriz de confusão."""
    # As predições estão em logits ou probabilidades, aplicar argmax para obter rótulos
    predicted_labels = np.argmax(all_predictions, axis=1)
    cm = confusion_matrix(all_labels, predicted_labels, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=targets)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    
    if data_val == "test" and test_folder is not None:
        cm_path = os.path.join(test_folder, "Confusion_matrix.png")
    else:
        cm_path = os.path.join(folder_path, "Confusion_matrix.png")
    plot_and_save_figure(fig, cm_path)

def plot_roc_curve(all_labels, all_predictions, num_classes, targets, folder_path, data_val="val", test_folder=None):
    """Gera e salva a curva ROC para classificação binária ou multiclasse."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if num_classes == 2:
        # Para classificação binária: usa a probabilidade da classe positiva
        y_scores = all_predictions[:, 1] if all_predictions.shape[1] == 2 else torch.sigmoid(torch.tensor(all_predictions)).numpy()
        fpr, tpr, _ = roc_curve(all_labels, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.6f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
        ax.set_title('Curva ROC - Classificação Binária', fontsize=16)
    else:
        # Para multiclasse: binariza os rótulos e gera curva ROC para cada classe
        all_labels_binarized = label_binarize(all_labels, classes=range(num_classes))
        y_scores = all_predictions if all_predictions.shape[1] == num_classes else torch.softmax(torch.tensor(all_predictions), dim=1).numpy()
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(all_labels_binarized[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'Classe {targets[i]} (AUC = {roc_auc:0.2f})')
        ax.set_title('Curva ROC - Classificação Multiclasse', fontsize=16)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=14)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    
    # Define caminho para salvar
    roc_path = os.path.join(test_folder if data_val=="test" and test_folder is not None else folder_path, "ROC_curve.png")
    plot_and_save_figure(fig, roc_path)

def save_model_and_metrics(model, metrics, model_name, base_dir, fold_num, all_labels, all_predictions, targets, data_val="val"):
    """
    Salva o modelo, as métricas, a matriz de confusão e a curva ROC em uma pasta específica.
    
    Args:
        model: Modelo treinado (torch.nn.Module).
        metrics: Dicionário com as métricas de avaliação.
        model_name: Nome do modelo.
        base_dir: Diretório base para salvar os resultados.
        fold_num: Número do fold (para validação cruzada).
        all_labels: Rótulos verdadeiros.
        all_predictions: Predições do modelo (probabilidades ou logits).
        targets: Lista de rótulos para exibição na matriz de confusão.
        data_val: Tipo de dados ('val' ou 'test').
    """
    # Converte listas para array NumPy se necessário
    all_predictions = ensure_numpy(all_predictions, "all_predictions")
    all_labels = ensure_numpy(all_labels, "all_labels")

    # Cria a pasta de resultados
    folder_path = create_folder(base_dir, model_name, fold_num)

    # Salva o modelo
    model_path = os.path.join(folder_path, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo em: {model_path}")

    # Salva as métricas em um CSV no diretório base
    metrics_file = os.path.join(base_dir, "model_metrics.csv")
    save_csv(metrics, metrics_file)

    # Se for dados de teste, cria pasta separada
    test_folder = None
    if data_val == "test":
        test_folder = os.path.join(base_dir, "test_results")
        os.makedirs(test_folder, exist_ok=True)

    # Plota e salva a matriz de confusão
    plot_confusion_matrix(all_labels, all_predictions, targets, folder_path, data_val, test_folder)

    # Determina o número de classes
    num_classes = all_predictions.shape[1] if len(all_predictions.shape) > 1 else 1

    # Plota e salva a curva ROC (para binária ou multiclasse)
    if num_classes >= 1:
        plot_roc_curve(all_labels, all_predictions, num_classes, targets, folder_path, data_val, test_folder)
    else:
        print("Número de classes não suportado para a geração da curva ROC.")

    return folder_path