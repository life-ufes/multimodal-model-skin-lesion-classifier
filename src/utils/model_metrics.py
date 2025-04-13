from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
import torch

def evaluate_model(model, dataloader, device, fold_num):
    model.eval()  # Coloca o modelo em modo de avaliação
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, metadata, labels in dataloader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            outputs = model(images, metadata)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            # Coleta os rótulos, previsões e probabilidades
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Converte as listas em arrays do NumPy para facilitar o cálculo
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calcula as métricas clássicas
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1score = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

    # Cálculo da AUC
    try:
        # Se for classificação binária, use somente a probabilidade da classe positiva
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        else:
            # Caso seja multiclasses, pode-se usar multi_class='ovr'
            auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
    except Exception as e:
        print(f"Erro ao processar o cálculo da AUC. Erro: {e}\n")
        auc = 0.0

    return {
        "fold": fold_num, 
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1score,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }, all_labels, all_predictions
