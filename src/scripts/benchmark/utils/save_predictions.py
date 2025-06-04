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

def model_val_predictions(model, dataloader, targets, device: str, fold_num: int, base_dir: str):
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
