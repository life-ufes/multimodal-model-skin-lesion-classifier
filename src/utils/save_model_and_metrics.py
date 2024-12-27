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
    # Converter all_predictions para um array NumPy se for uma lista
    if isinstance(all_predictions, list):
        all_predictions = np.array(all_predictions)
        print("Conversão de all_predictions de lista para array NumPy realizada.")

    # Garantir que all_labels também esteja como array NumPy
    if isinstance(all_labels, list):
        all_labels = np.array(all_labels)
        print("Conversão de all_labels de lista para array NumPy realizada.")

    # Criar o diretório base se não existir
    os.makedirs(base_dir, exist_ok=True)

    # Gerar o nome único da pasta usando o nome do modelo e o timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_name}_fold_{fold_num}_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)

    # Criar a pasta para o modelo
    os.makedirs(folder_path, exist_ok=True)

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

    # Calcular matriz de confusão
    predicted_labels = np.argmax(all_predictions, axis=1)
    cm = confusion_matrix(all_labels, predicted_labels, normalize='true')

    # Salvar a matriz de confusão como gráfico
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=targets)
    cm_display.plot(cmap=plt.cm.Blues)

    # Determinar o caminho para salvar a matriz de confusão
    if data_val == "test":
        test_folder = os.path.join(base_dir, "test_results")
        os.makedirs(test_folder, exist_ok=True)
        cm_path = os.path.join(test_folder, "Confusion_matrix.png")
    else:
        cm_path = os.path.join(folder_path, "Confusion_matrix.png")

    # Salvar a figura localmente
    plt.savefig(cm_path)
    plt.close()  # Fecha a figura para liberar memória
    print(f"Matriz de confusão salva em: {cm_path}")

    # --- Início da Geração e Salvamento da Curva ROC ---

    # Determinar o número de classes
    if len(all_predictions.shape) > 1:
        num_classes = all_predictions.shape[1]
    else:
        num_classes = 1  # Não suportado

    # Classificação Binária
    if num_classes == 2:
        # Garantir que as predições são probabilidades da classe positiva
        if all_predictions.shape[1] == 2:
            y_scores = all_predictions[:, 1]
        else:
            # Aplicar sigmoid se necessário
            y_scores = torch.sigmoid(torch.tensor(all_predictions)).numpy()

        # Calcular a curva ROC
        fpr, tpr, thresholds = roc_curve(all_labels, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plotar a curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=14)
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=14)
        plt.title('Curva ROC - Classificação Binária', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)

        # Determinar o caminho para salvar a curva ROC
        if data_val == "test":
            roc_path = os.path.join(test_folder, "ROC_curve.png")
        else:
            roc_path = os.path.join(folder_path, "ROC_curve.png")

        # Salvar a figura localmente
        plt.savefig(roc_path)
        plt.close()
        print(f"Curva ROC salva em: {roc_path}")

    # Classificação Multiclasse
    elif num_classes > 2:
        # Binarizar os rótulos para multiclasse
        all_labels_binarized = label_binarize(all_labels, classes=range(num_classes))
        # Verificar se as predições são probabilidades
        if all_predictions.shape[1] == num_classes:
            y_scores = all_predictions
        else:
            # Aplicar softmax se necessário
            y_scores = torch.softmax(torch.tensor(all_predictions), dim=1).numpy()

        # Calcular a curva ROC para cada classe
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels_binarized[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plotar todas as curvas ROC
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)],
                     lw=2, label=f'Classe {targets[i]} (AUC = {roc_auc[i]:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=14)
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=14)
        plt.title('Curva ROC - Classificação Multiclasse', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)

        # Determinar o caminho para salvar a curva ROC
        if data_val == "test":
            roc_path = os.path.join(test_folder, "ROC_curve.png")
        else:
            roc_path = os.path.join(folder_path, "ROC_curve.png")

        # Salvar a figura localmente
        plt.savefig(roc_path)
        plt.close()
        print(f"Curva ROC salva em: {roc_path}")

    else:
        print("Número de classes não suportado para a geração da curva ROC.")

    # --- Fim da Geração e Salvamento da Curva ROC ---

    return folder_path
