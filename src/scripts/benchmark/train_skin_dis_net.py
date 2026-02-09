"""
train_skindisnet_multimodal.py

Treino multimodal (imagem + metadados) no SkinDisNet (MULTICLASS)
- Labels: Diagnosis
- Group split: Patient_id (StratifiedGroupKFold)
- Encoders (OHE + LabelEncoder) treinados APENAS no treino de cada fold (fold-aware)
- WeightedRandomSampler
- FocalLoss
- EarlyStopping por val_bacc (balanced_accuracy)
- MLflow logging
- Compatível com: MDNet, LiwTERM, MetaNetModel, MultimodalModel

OBS:
- Este script assume que o SkinDisNetDataset exista (ou você pode colar a classe aqui).
- Ele cria preprocess_dir por fold: ./data/preprocess_data/skindisnet/fold_{k}/ (onde ficam ohe + le).
- Se houver imagem faltando, o dataset pode retornar None (e o collate filtra).
"""

import os
import time
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import mlflow

from sklearn.model_selection import StratifiedGroupKFold

# ===== Seus imports do projeto =====
from utils import model_metrics, save_predictions, load_local_variables
from utils.early_stopping import EarlyStopping
from utils.save_model_and_metrics import save_model_and_metrics

import models.focalLoss as focalLoss
from models import multimodalIntraInterModal
from models.multimodalMDNet import MDNet
from models.liwtermModel import LiwTERM
from models.metanet import MetaNetModel

# >>> Ajuste aqui para o seu módulo do dataset:
# from models.skinLesionDatasetsSkinDisNet import SkinDisNetDataset
from models import skinLesionDatasetsSkinDisNet  # deve conter SkinDisNetDataset


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def compute_class_weights(labels, num_classes):
    """
    Inverse-frequency weights: total / (C * count_c)
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    weights = []
    for i in range(num_classes):
        if counts[i] > 0:
            weights.append(total / (num_classes * counts[i]))
        else:
            weights.append(0.0)
    return torch.tensor(weights, dtype=torch.float)


def ensure_fold_preprocess_dir(preprocess_root: str, fold_num: int) -> str:
    fold_dir = os.path.join(preprocess_root, f"fold_{fold_num}")
    shutil.rmtree(fold_dir, ignore_errors=True)
    os.makedirs(fold_dir, exist_ok=True)
    return fold_dir


def skindisnet_collate(batch):
    """
    Remove amostras None (ex.: imagem faltando ou label inválido)
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# -----------------------------------------------------------------------------
# Train loop
# -----------------------------------------------------------------------------
def train_process(
    num_epochs,
    num_heads,
    fold_num,
    train_loader,
    val_loader,
    targets,
    model,
    device,
    weightes_per_category,
    common_dim,
    model_name,
    text_model_encoder,
    attention_mecanism,
    results_folder_path
):
    # Loss: FocalLoss com pesos por classe
    # criterion = focalLoss.FocalLoss(
    #     alpha=weightes_per_category,
    #     gamma=2.0,
    #     reduction="mean"
    # )
    criterion = nn.CrossEntropyLoss(weight=weightes_per_category)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
        verbose=True
    )

    model.to(device)

    model_save_path = os.path.join(
        results_folder_path,
        f"model_{model_name}_with_{text_model_encoder}_{common_dim}_with_best_architecture"
    )
    os.makedirs(model_save_path, exist_ok=True)
    print(model_save_path)

    early_stopping = EarlyStopping(
        patience=10,
        delta=0.01,
        verbose=True,
        path=str(model_save_path + f'/{model_name}_fold_{fold_num}/best-model/'),
        save_to_disk=False,
        early_stopping_metric_name="val_bacc"
    )

    initial_time = time.time()
    train_losses, val_losses = [], []
    epoch_index = 0

    experiment_name = "EXPERIMENTOS-SKINDISNET - RESIDUAL BLOCK USAGE - 2026-01-16"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"image_extractor_model_{model_name}_with_mecanism_{attention_mecanism}_fold_{fold_num}_num_heads_{num_heads}"
    ):
        mlflow.log_param("fold_num", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mecanism", attention_mecanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", "focal_loss")
        mlflow.log_param("num_heads", num_heads)

        for epoch_index in range(num_epochs):
            # -----------------------
            # Train
            # -----------------------
            model.train()
            running_loss = 0.0
            n_batches = 0

            for batch in tqdm(
                train_loader, desc=f"[Fold {fold_num}] Epoch {epoch_index+1}/{num_epochs}", leave=False
            ):
                if batch is None:
                    continue
                _, image, metadata, label = batch
                image = image.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / max(1, n_batches)
            print(f"\n[Fold {fold_num}] Training: Epoch {epoch_index}, Loss: {train_loss:.4f}")

            # -----------------------
            # Validation
            # -----------------------
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    _, image, metadata, label = batch
                    image = image.to(device, non_blocking=True)
                    metadata = metadata.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)

                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss = val_loss / max(1, n_val_batches)
            print(f"[Fold {fold_num}] Validation Loss: {val_loss:.4f}")

            scheduler.step(val_loss)
            current_lr = [pg["lr"] for pg in optimizer.param_groups]
            print(f"[Fold {fold_num}] Current Learning Rate(s): {current_lr}\n")

            metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                fold_num=fold_num,
                targets=targets,
                base_dir=model_save_path,
                model_name=model_name
            )

            metrics["epoch"] = int(epoch_index)
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)

            print(f"[Fold {fold_num}] Metrics: {metrics}")

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, np.floating)):
                    if np.isnan(metric_value):
                        continue
                    mlflow.log_metric(metric_name, float(metric_value), step=epoch_index + 1)
                else:
                    mlflow.log_param(metric_name, str(metric_value))

            early_stopping(val_loss=val_loss, val_bacc=float(metrics["balanced_accuracy"]), model=model)
            if early_stopping.early_stop:
                print(f"[Fold {fold_num}] Early stopping triggered!")
                break

            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))

    train_process_time = time.time() - initial_time

    # Carrega o melhor modelo encontrado
    model = early_stopping.load_best_weights(model)
    model.eval()

    # Inferência final com o melhor modelo
    with torch.no_grad():
        metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            fold_num=fold_num,
            targets=targets,
            base_dir=model_save_path,
            model_name=model_name
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
        val_losses=val_losses
    )

    print(f"[Fold {fold_num}] Model saved at {model_save_path}")
    return model, model_save_path


# -----------------------------------------------------------------------------
# Pipeline (StratifiedGroupKFold por Patient_id)
# -----------------------------------------------------------------------------
def pipeline(
    base_dataset,
    num_epochs,
    batch_size,
    device,
    k_folds,
    model_name,
    num_heads,
    common_dim,
    text_model_encoder,
    unfreeze_weights,
    attention_mecanism,
    results_folder_path,
    num_workers=10,
    persistent_workers=True,
    preprocess_root="./data/preprocess_data/skindisnet",
):
    """
    base_dataset: SkinDisNetDataset com fit_encoders=False, preprocess_dir qualquer, e metadata carregada.
                 Importante: este dataset base NÃO deve tentar carregar ohe/le do disco.
    """

    # labels e groups do dataset base (antes de treinar encoders por fold)
    # Aqui assumimos que o dataset base já tem:
    # - base_dataset.metadata com colunas Patient_id e Diagnosis
    # - base_dataset.labels (LabelEncoder global ou provisório) só para split
    # Se você preferir, dá para gerar labels provisórios aqui via pandas factorize.
    if getattr(base_dataset, "labels", None) is None or len(base_dataset.labels) != len(base_dataset):
        # fallback seguro: labels provisórios só para o split
        base_labels, _ = pd.factorize(base_dataset.metadata["Diagnosis"].astype(str).values)
        labels_for_split = base_labels
    else:
        labels_for_split = np.asarray(base_dataset.labels, dtype=int)

    groups = base_dataset.metadata["Patient_id"].astype(str).values

    skf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_for_split)), labels_for_split, groups)):
        fold_num = fold + 1
        print("\n==============================")
        print(f"Fold {fold_num}/{k_folds}")
        print("==============================")

        train_labels_split = labels_for_split[train_idx].tolist()
        val_labels_split = labels_for_split[val_idx].tolist()
        print(f"Fold {fold_num}: train={Counter(train_labels_split)}, val={Counter(val_labels_split)}")

        # preprocess dir do fold (encoders treinados só no treino)
        fold_dir = ensure_fold_preprocess_dir(preprocess_root, fold_num)

        # -----------------------
        # Datasets do fold (TREINA encoders no treino)
        # -----------------------
        train_dataset = skinLesionDatasetsSkinDisNet.SkinDisNetDataset(
            csv_file=base_dataset.csv_file,
            img_root=base_dataset.img_root,
            size=base_dataset.size,
            is_train=False,
            preprocess_dir=fold_dir,
            fit_encoders=True,   # << treina OHE+LE aqui
        )
        train_dataset.metadata = base_dataset.metadata.iloc[train_idx].reset_index(drop=True)
        train_dataset.features, train_dataset.labels, train_dataset.targets = train_dataset._process_metadata()

        val_dataset = skinLesionDatasetsSkinDisNet.SkinDisNetDataset(
            csv_file=base_dataset.csv_file,
            img_root=base_dataset.img_root,
            size=base_dataset.size,
            is_train=False,
            preprocess_dir=fold_dir,
            fit_encoders=False,  # << usa OHE+LE do treino
        )
        val_dataset.metadata = base_dataset.metadata.iloc[val_idx].reset_index(drop=True)

        # Se aparecer label "unseen" no val (raríssimo, mas pode), a estratégia mais segura é dropar esses itens.
        # A sua classe pode implementar isso dentro de _process_metadata; aqui só chamamos.
        val_dataset.features, val_dataset.labels, val_dataset.targets = val_dataset._process_metadata()

        # dims
        num_metadata_features = int(train_dataset.features.shape[1])
        num_classes = int(len(train_dataset.targets))
        targets = list(train_dataset.targets)

        print(f"[Fold {fold_num}] num_metadata_features={num_metadata_features} | num_classes={num_classes}")

        # -----------------------
        # Sampler balanceado
        # -----------------------
        class_weights = compute_class_weights(train_dataset.labels, num_classes).to(device)
        print(f"[Fold {fold_num}] Pesos das classes: {class_weights}")

        sample_weights = torch.tensor([class_weights[y].item() for y in train_dataset.labels], dtype=torch.float)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # -----------------------
        # Loaders
        # -----------------------
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=skindisnet_collate,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=skindisnet_collate,
        )

        # -----------------------
        # Modelo multimodal
        # -----------------------
        if attention_mecanism == "md-net":
            model = MDNet(
                meta_dim=num_metadata_features,
                num_classes=num_classes,
                unfreeze_weights=unfreeze_weights,
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
                unfreeze_weights=unfreeze_weights,
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
                unfreeze_weights=unfreeze_weights,
                attention_mecanism=attention_mecanism,
                n=1 if attention_mecanism == "no-metadata" else 2
            )

        # -----------------------
        # Treino
        # -----------------------
        model, model_save_path = train_process(
            num_epochs=num_epochs,
            num_heads=num_heads,
            fold_num=fold_num,
            train_loader=train_loader,
            val_loader=val_loader,
            targets=targets,
            model=model,
            device=device,
            weightes_per_category=class_weights,
            common_dim=common_dim,
            model_name=model_name,
            text_model_encoder=text_model_encoder,
            attention_mecanism=attention_mecanism,
            results_folder_path=results_folder_path,
        )

        # -----------------------
        # Salvar predições
        # -----------------------
        save_predictions.model_val_predictions(
            model=model,
            dataloader=val_loader,
            device=device,
            fold_num=fold_num,
            targets=targets,
            base_dir=model_save_path,
            model_name=model_name
        )


# -----------------------------------------------------------------------------
# Entrypoint (estilo PAD-20)
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
        "full_unfrozen": "unfrozen_weights",
        "partial_unfrozen": "partial_weights",
        "totally_frozen": "frozen_weights"
    }
    train_mode_folder = TRAIN_MODE_FOLDER.get(unfreeze_weights, "frozen_weights")
    results_folder_path = f"{results_folder_path}/{dataset_folder_name}/{train_mode_folder}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model_encoder = "one-hot-encoder"

    list_of_attention_mecanism = ["att-intramodal+residual+cross-attention-metadados"]
    list_of_models = ["mobilenet-v2", "davit_tiny.msft_in1k", "mvitv2_small.fb_in1k", "coat_lite_small.in1k", "caformer_b36.sail_in22k_ft_in1k", "vgg16", "densenet169", "resnet-50"]
    list_num_heads = [8] if list_num_heads is None else list_num_heads

    # >>> Ajuste os caminhos do SkinDisNet aqui:
    # Ex.: dataset_folder_path = "/data/SkinDisNet"
    # Deve existir um CSV com as colunas: Folder_name,Patient_id,Image_id,Age,Sex,Leision_location,Diagnosis
    csv_file = os.path.join(dataset_folder_path, "SkinDisNet_Metadata.csv")  # ou o nome real do seu csv
    img_root = os.path.join(dataset_folder_path, "Preprocessed")        # ou raiz que contém as subpastas "Folder_name"

    # Base dataset: NÃO treina encoders aqui (para não exigir ohe/le no disco)
    base_dataset = skinLesionDatasetsSkinDisNet.SkinDisNetDataset(
        csv_file=csv_file,
        img_root=img_root,
        size=(224, 224),
        is_train=True,
        preprocess_dir="./data/preprocess_data/skindisnet/base",
        fit_encoders=False,
        build_features=False   # <<< CRÍTICO
    )

    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    pipeline(
                        base_dataset=base_dataset,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        device=device,
                        k_folds=k_folds,
                        model_name=model_name,
                        num_heads=int(num_heads),
                        common_dim=common_dim,
                        text_model_encoder=text_model_encoder,
                        unfreeze_weights=unfreeze_weights,
                        attention_mecanism=attention_mecanism,
                        results_folder_path=f"{results_folder_path}/SkinDisNet/{num_heads}/{attention_mecanism}",
                        num_workers=num_workers,
                        persistent_workers=True,
                        preprocess_root="./data/preprocess_data/skindisnet"
                    )
                except Exception as e:
                    print(f"Erro ao processar modelo={model_name}, mecanismo={attention_mecanism}: {e}")
                    continue


if __name__ == "__main__":
    run_experiments()
