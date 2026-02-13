"""
train_midas_multimodal.py

Treino multimodal (imagem + metadados) no dataset MIDAS com:
- StratifiedGroupKFold (lesion-wise) usando midas_record_id
- Encoders (OHE/Scaler/LabelEncoder) treinados APENAS no treino de cada fold
- DataLoaders com WeightedRandomSampler
- EarlyStopping por balanced_accuracy (val_bacc)
- MLflow logging
- Compatível com seus modelos (MDNet, LiwTERM, MetaNetModel, MultimodalModel)

Ajuste os imports dos seus módulos conforme o seu projeto.
"""

import os
import time
import shutil
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import mlflow

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

# ===== Seus imports do projeto =====
from utils import model_metrics, save_predictions, load_local_variables
from utils.early_stopping import EarlyStopping
from utils.save_model_and_metrics import save_model_and_metrics

import models.focalLoss as focalLoss
from models import multimodalIntraInterModal
from models.multimodalMDNet import MDNet
from models.liwtermModel import LiwTERM
from models.metanet import MetaNetModel

from models import skinLesionDatasetsMIDAS  # deve conter MIDASDataset atualizado


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    weights = []
    for i in range(num_classes):
        if counts[i] > 0:
            weight = total_samples / (num_classes * counts[i])
        else:
            weight = 0.0
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float)


def ensure_fold_preprocess_dir(base_dir: str, fold_num: int) -> str:
    fold_dir = os.path.join(base_dir, f"fold_{fold_num}")
    shutil.rmtree(fold_dir, ignore_errors=True)
    os.makedirs(fold_dir, exist_ok=True)
    return fold_dir

def midas_collate(batch):
    # remove amostras None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# -----------------------------------------------------------------------------
# Train loop
# -----------------------------------------------------------------------------
def train_process(
    num_epochs: int,
    num_heads: int,
    fold_num: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    targets,
    model,
    device,
    class_weights: torch.Tensor,
    common_dim: int,
    model_name: str,
    text_model_encoder: str,
    attention_mecanism: str,
    results_folder_path: str,
):
    criterion = focalLoss.FocalLoss(
        alpha=class_weights,
        gamma=2.0,
        reduction="mean",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    model.to(device)

    model_save_path = os.path.join(
        results_folder_path,
        f"model_{model_name}_with_{text_model_encoder}_{common_dim}_with_best_architecture",
    )
    os.makedirs(model_save_path, exist_ok=True)

    early_stopping = EarlyStopping(
        patience=10,
        delta=0.01,
        verbose=True,
        path=str(os.path.join(model_save_path, f"{model_name}_fold_{fold_num}", "best-model")),
        save_to_disk=False,
        early_stopping_metric_name="val_bacc",
    )

    initial_time = time.time()
    train_losses, val_losses = [], []
    epoch_index = 0

    # Ajuste aqui o nome do experimento
    experiment_name = "EXPERIMENTOS-MIDAS - RESIDUAL BLOCK USAGE - 2026-01-01"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"image_extractor_{model_name}_{attention_mecanism}_fold_{fold_num}_heads_{num_heads}"
    ):
        mlflow.log_param("fold_num", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mecanism", attention_mecanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", "focal_loss")
        mlflow.log_param("num_heads", num_heads)
        mlflow.log_param("common_dim", common_dim)

        for epoch_index in range(num_epochs):
            # -------- Train --------
            model.train()
            running_loss = 0.0

            for _, image, metadata, label in tqdm(
                train_loader, desc=f"[Fold {fold_num}] Epoch {epoch_index+1}/{num_epochs}", leave=False
            ):
                if image is None:
                    continue
                image = image.to(device)
                metadata = metadata.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / max(1, len(train_loader))
            print(f"\n[Fold {fold_num}] Training: Epoch {epoch_index+1}, Loss: {train_loss:.4f}")

            # -------- Val --------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _, image, metadata, label in val_loader:
                    image = image.to(device)
                    metadata = metadata.to(device)
                    label = label.to(device)

                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()

            val_loss = val_loss / max(1, len(val_loader))
            print(f"[Fold {fold_num}] Validation Loss: {val_loss:.4f}")

            scheduler.step(val_loss)
            current_lr = [pg["lr"] for pg in optimizer.param_groups]
            print(f"[Fold {fold_num}] Current LR(s): {current_lr}\n")

            metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                fold_num=fold_num,
                targets=targets,
                base_dir=model_save_path,
                model_name=model_name,
            )

            metrics["epoch"] = epoch_index
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)

            print(f"[Fold {fold_num}] Metrics: {metrics}")

            # MLflow log
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, np.floating)):
                    if np.isnan(metric_value):
                        continue
                    mlflow.log_metric(metric_name, float(metric_value), step=epoch_index + 1)
                else:
                    mlflow.log_param(metric_name, str(metric_value))

            # Early stopping (usa val_bacc)
            early_stopping(
                val_loss=val_loss,
                val_bacc=float(metrics.get("balanced_accuracy", 0.0)),
                model=model,
            )
            if early_stopping.early_stop:
                print(f"[Fold {fold_num}] Early stopping triggered!")
                break

            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))

    train_process_time = time.time() - initial_time

    # Carrega o melhor modelo
    model = early_stopping.load_best_weights(model)
    model.eval()

    # Inferência final na validação (melhor modelo)
    with torch.no_grad():
        metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            fold_num=fold_num,
            targets=targets,
            base_dir=model_save_path,
            model_name=model_name,
        )

    metrics["train process time"] = str(train_process_time)
    metrics["epochs"] = str(int(epoch_index))
    metrics["data_val"] = "val"

    save_model_and_metrics(
        model=model,
        metrics=metrics,
        model_name=model_name,
        base_dir=model_save_path,
        save_to_disk=True,
        fold_num=fold_num,
        all_labels=all_labels,
        all_predictions=all_predictions,
        all_probabilities=all_probs,
        targets=targets,
        data_val="val",
        train_losses=train_losses,
        val_losses=val_losses,
    )

    print(f"[Fold {fold_num}] Model saved at {model_save_path}")

    return model, model_save_path


# -----------------------------------------------------------------------------
# Pipeline (StratifiedGroupKFold lesion-wise)
# -----------------------------------------------------------------------------
def pipeline(
    dataset,  # MIDASDataset com build_features=False (apenas metadata)
    num_epochs: int,
    batch_size: int,
    device,
    k_folds: int,
    model_name: str,
    num_heads: int,
    common_dim: int,
    text_model_encoder: str,
    unfreeze_weights,
    attention_mecanism: str,
    results_folder_path: str,
    num_workers: int = 10,
    persistent_workers: bool = True,
    preprocess_root: str = "./data/preprocess_data/midas",
):
    # labels e groups (lesion-wise)
    labels = (
        dataset.metadata["midas_path"]
        .astype(str)
        .str.lower()
        .str.startswith("malignant")
        .astype(int)
        .values
    )

    groups = dataset.metadata["midas_record_id"].values
    targets = dataset.targets
    num_classes = len(np.unique(labels))   # deve dar 2

    stratifiedKFold = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(stratifiedKFold.split(range(len(dataset)), labels, groups=groups)):
        print(f"\n==============================")
        print(f"Fold {fold+1}/{k_folds}")
        print(f"==============================")

        # distribuição das classes no fold
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        print(f"Fold {fold}: train={Counter(train_labels)}, val={Counter(val_labels)}")

        # preprocess dir do fold
        fold_dir = ensure_fold_preprocess_dir(preprocess_root, fold+1)

        # -------- Datasets do fold (SEM leakage) --------
        train_dataset = skinLesionDatasetsMIDAS.MIDASDataset(
            metadata_file=dataset.metadata_file,
            img_dir=dataset.img_dir,
            size=dataset.size,
            is_train=True,
            preprocess_dir=fold_dir,
            fit_encoders=True,
            build_features=False,  # evita processar antes do slice
        )
        train_dataset.metadata = dataset.metadata.iloc[train_idx].reset_index(drop=True)
        train_dataset.features, train_dataset.labels, train_dataset.targets = train_dataset._process_metadata()

        val_dataset = skinLesionDatasetsMIDAS.MIDASDataset(
            metadata_file=dataset.metadata_file,
            img_dir=dataset.img_dir,
            size=dataset.size,
            is_train=False,
            preprocess_dir=fold_dir,
            fit_encoders=False,
            build_features=False,
        )
        val_dataset.metadata = dataset.metadata.iloc[val_idx].reset_index(drop=True)
        val_dataset.features, val_dataset.labels, _ = val_dataset._process_metadata()
        val_dataset.targets = targets  # garante targets consistentes

        # número de features tabulares (por fold, consistente)
        num_metadata_features = train_dataset.features.shape[1]
        print(f"[Fold {fold+1}] num_metadata_features={num_metadata_features} | num_classes={num_classes}")

        # -------- Sampler balanceado --------
        class_weights = compute_class_weights(train_labels, num_classes).to(device)
        print(f"[Fold {fold+1}] Class weights: {class_weights}")

        sample_weights = torch.tensor([class_weights[y].item() for y in train_labels], dtype=torch.float)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # -------- Loaders --------
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=midas_collate
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=midas_collate
        )

        # -------- Modelo --------
        if attention_mecanism == "md-net":
            model = MDNet(
                meta_dim=num_metadata_features,
                num_classes=num_classes,
                unfreeze_weights=status_weights,
                cnn_model_name=model_name,
                device=device,
            )
        elif attention_mecanism == "liwterm":
            model = LiwTERM(
                num_classes=num_classes,
                meta_dim=num_metadata_features,
                image_encoder="vit_large_patch16_224",
                pretrained=True,
                unfreeze_backbone=unfreeze_weights,
            )
        elif attention_mecanism == "metanet":
            model = MetaNetModel(
                meta_dim=num_metadata_features,
                num_classes=num_classes,
                image_encoder=str(model_name).replace("-", ""),
                unfreeze_weights=status_weights,
            )
        else:
            model = multimodalIntraInterModal.MultimodalModel(
                num_classes=num_classes,
                num_heads=num_heads,
                device=device,
                cnn_model_name=model_name,
                text_model_name=text_model_encoder,
                common_dim=common_dim,
                vocab_size=num_metadata_features,
                unfreeze_weights=status_weights,
                attention_mecanism=attention_mecanism,
                n=1 if attention_mecanism == "no-metadata" else 2,
            )

        # -------- Treino --------
        model, model_save_path = train_process(
            num_epochs=num_epochs,
            num_heads=num_heads,
            fold_num=fold,
            train_loader=train_loader,
            val_loader=val_loader,
            targets=targets,
            model=model,
            device=device,
            class_weights=class_weights,
            common_dim=common_dim,
            model_name=model_name,
            text_model_encoder=text_model_encoder,
            attention_mecanism=attention_mecanism,
            results_folder_path=results_folder_path,
        )

        # -------- Salvar predições --------
        save_predictions.model_val_predictions(
            model=model,
            dataloader=val_loader,
            device=device,
            fold_num=fold,
            targets=targets,
            base_dir=model_save_path,
            model_name=model_name,
        )


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def run_experiments():
    local_variables = load_local_variables.get_env_variables()

    num_epochs = int(local_variables["num_epochs"])
    batch_size = int(local_variables["batch_size"])
    k_folds = int(local_variables["k_folds"])
    common_dim = int(local_variables["common_dim"])
    list_num_heads = local_variables["list_num_heads"]
    num_workers = int(local_variables["num_workers"])
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = local_variables["dataset_folder_path"]
    unfreeze_weights = str(local_variables["unfreeze_weights"])
    results_folder_path = str(local_variables["results_folder_path"])
    TRAIN_MODE_FOLDER = {
        "unfrozen_weights": "unfrozen_weights",
        "last_layer_unfrozen_weights": "partial_weights",
        "frozen_weights": "frozen_weights"
    }
    train_mode_folder = TRAIN_MODE_FOLDER.get(unfreeze_weights, "frozen_weights")
    results_folder_path = f"{results_folder_path}/{dataset_folder_name}/{train_mode_folder}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model_encoder = "one-hot-encoder"

    list_of_attention_mecanism = ["att-intramodal+residual+cross-attention-metadados"]
    list_of_models = ["mobilenet-v2", "davit_tiny.msft_in1k", "mvitv2_small.fb_in1k", "coat_lite_small.in1k",
                      "caformer_b36.sail_in22k_ft_in1k", "vgg16", "densenet169", "resnet-50"]

    # Dataset base (apenas para split)
    metadata_file = os.path.join(dataset_folder_path, "release_midas.xlsx")
    img_dir = os.path.join(dataset_folder_path, "images")  # ajuste se necessário

    dataset = skinLesionDatasetsMIDAS.MIDASDataset(
        metadata_file=metadata_file,
        img_dir=img_dir,
        size=(224, 224),
        is_train=True,
        preprocess_dir="./data/preprocess_data/midas",
        fit_encoders=False,
        build_features=False
    )

    # labels para o split (sem OHE/Scaler)
    le = LabelEncoder()
    dataset.labels = le.fit_transform(dataset.metadata["midas_path"].astype(str).values)
    dataset.targets = le.classes_

    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    pipeline(
                        dataset=dataset,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        device=device,
                        k_folds=k_folds,
                        model_name=model_name,
                        num_heads=int(num_heads),
                        common_dim=common_dim,
                        text_model_encoder=text_model_encoder,
                        unfreeze_weights=status_weights,
                        attention_mecanism=attention_mecanism,
                        results_folder_path=f"{results_folder_path}/{num_heads}/{attention_mecanism}",
                        num_workers=num_workers,
                        persistent_workers=True,
                        preprocess_root="./data/preprocess_data/midas",
                    )
                except Exception as e:
                    print(f"Erro ao processar modelo={model_name}, mecanismo={attention_mecanism}: {e}")
                    continue


if __name__ == "__main__":
    run_experiments()
