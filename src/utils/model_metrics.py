from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
import torch

def evaluate_model(model, dataloader, device):
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

            # Coletar os rótulos e previsões
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)

    # AUC - Somente para problemas binários ou multi-class com probabilidades
    try:
        auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
    except ValueError:  # Caso AUC não possa ser calculado
        auc = None

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }
