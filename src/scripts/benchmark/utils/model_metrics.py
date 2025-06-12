from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
import os
import pandas as pd

def evaluate_model(model, dataloader, targets, device: str, fold_num: int, base_dir: str, model_name:str="None"):
    # Gerar o nome único da pasta usando o nome do modelo e o timestamp
    folder_name = f"{model_name}_fold_{fold_num}"
    folder_path = os.path.join(base_dir, folder_name)

    # Criar a pasta para o modelo
    os.makedirs(folder_path, exist_ok=True)

    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for (image_names, images, metadata, labels) in dataloader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            outputs = model(images, metadata)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    targets = np.array(targets)

    # Salvamento de arrays
    np.save(os.path.join(folder_path, "labels.npy"), all_labels)
    np.save(os.path.join(folder_path, "predictions.npy"), all_predictions)
    np.save(os.path.join(folder_path, "targets.npy"), targets)

    # Métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1score = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    try:
        classes = np.unique(all_labels)
        if len(classes) == 2:
            auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        else:
            y_true_bin = label_binarize(all_labels, classes=classes)
            auc = roc_auc_score(y_true_bin, all_probabilities, average='macro', multi_class='ovr')
    except Exception as e:
        print(f"Erro ao calcular AUC: {e}")
        auc = None

    return {
        "fold": fold_num,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1score,
        "auc": auc
    }, all_labels, all_probabilities
