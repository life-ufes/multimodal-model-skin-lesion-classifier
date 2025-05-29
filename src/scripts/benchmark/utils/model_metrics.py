from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
import os
import pandas as pd
def save_image_prediction_in_evaluation_by_fold(
    file_folder_path: str,
    fold_num: int,
    image_names,
    labels,
    preds,
    probs,
    targets  # Lista com os nomes das classes, ex: ['NEV', 'BCC', 'ACK', ...]
):
    try:
        image_names = list(image_names)
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
        preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds)
        probs = probs.cpu().numpy() if isinstance(probs, torch.Tensor) else np.array(probs)

        n = len(image_names)
        if not all(len(x) == n for x in [labels, preds, probs]):
            print(f"Tamanhos incompatíveis:")
            print(f"image_names: {len(image_names)}, labels: {len(labels)}, preds: {len(preds)}, probs: {len(probs)}")
            return

        # Mapeamento para nomes das classes
        label_names = [targets[i] for i in labels]
        pred_names = [targets[i] for i in preds]

        # Criar DataFrame base
        df = pd.DataFrame({
            "image_name": image_names,
            "label_idx": labels,
            "label": label_names,
            "prediction_idx": preds,
            "prediction": pred_names
        })

        # Adicionar probabilidade de cada classe como coluna
        for class_idx, class_name in enumerate(targets):
            df[f"prob_{class_name}"] = probs[:, class_idx]

        csv_path = os.path.join(file_folder_path, f"predictions_eval_fold_{fold_num}.csv")
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', header=write_header, index=False)
    except Exception as e:
        print(f"Erro ao tentar salvar as predições! Erro: {e}\n")

def evaluate_model(model, dataloader, targets, device: str, fold_num: int, base_dir: str):
    folder_path = os.path.join(base_dir, str(fold_num))
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

            save_image_prediction_in_evaluation_by_fold(
                file_folder_path=folder_path,
                fold_num=fold_num,
                targets=targets,
                image_names=image_names,
                preds=preds,
                probs=probs,
                labels=labels
            )

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
