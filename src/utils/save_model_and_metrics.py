import os
import csv
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_model_and_metrics(model, metrics, model_name, base_dir, fold_num, all_labels, all_predictions, targets):
    """
    Salva o modelo e as métricas em uma pasta específica.

    Args:
        model: O modelo treinado (torch.nn.Module).
        metrics: Dicionário contendo as métricas de avaliação.
        model_name: Nome do modelo.
        base_dir: Diretório base onde as pastas serão criadas.
    """
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
    file_exists = os.path.isfile(f"{base_dir}/model_metrics.csv")
    with open(f"{base_dir}/model_metrics.csv", mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())

        # Escrever cabeçalho apenas se o arquivo for criado agora
        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)
 
    # Calcular matriz de confusão
    cm = confusion_matrix(all_labels, np.argmax(all_predictions, axis=1), normalize='true')

    # Salvar a matriz de confusão como gráfico
    cm_display = ConfusionMatrixDisplay(cm, display_labels=targets)
    cm_display.plot(cmap=plt.cm.Blues)
    cm_path = os.path.join(folder_path, f"Confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()  # Fecha a figura para liberar memória
    print(f"Matriz de confusão salva em: {cm_path}")

    return folder_path
