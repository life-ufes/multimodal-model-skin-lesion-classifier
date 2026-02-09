#!/usr/bin/env python3
# ============================================================
# MILK10K - Multimodal training script (paper-safe CV)
# Fixes:
#  - Correct StratifiedKFold application (Subset)
#  - Train/Val datasets with different transforms (is_train True/False)
#  - Safe class weights (always num_classes length)
#  - Choose ONE imbalance strategy (recommended defaults below)
#  - Scheduler aligned with balanced_accuracy
#  - EarlyStopping aligned and less aggressive
# ============================================================

import os
import time
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from models.multimodalMDNet import MDNet

import mlflow
from tqdm import tqdm

# ------------------------------------------------------------
# Your project imports
# ------------------------------------------------------------
from utils import model_metrics, save_predictions
from utils.save_model_and_metrics import save_model_and_metrics
from utils import load_local_variables
from models import multimodalIntraInterModal
from models import skinLesionDatasetsMILK10K


# ============================================================
# CONFIG: choose ONE imbalance strategy (recommended)
# ============================================================

# Strategy A (recommended first): sampler + CrossEntropy (no class weights)
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTS_IN_LOSS = False
USE_FOCAL_LOSS = False

# Strategy B: FocalLoss (alpha capped) + no sampler
# USE_WEIGHTED_SAMPLER = False
# USE_CLASS_WEIGHTS_IN_LOSS = False
# USE_FOCAL_LOSS = True

FOCAL_GAMMA = 1.5
ALPHA_SQRT = True          # alpha := sqrt(weights)
ALPHA_CLAMP_MIN = 0.25
ALPHA_CLAMP_MAX = 5.0      # keep it sane


# ============================================================
# Utilities
# ============================================================

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Always returns a vector of length num_classes.
    Handles folds where some classes may be absent.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss on logits.
    alpha: Tensor[num_classes] or None
    gamma: float
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B]
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)  # pt = softmax prob of true class

        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================
# Training
# ============================================================

def train_process(
    num_epochs,
    num_heads,
    fold_num,
    train_loader,
    val_loader,
    targets,
    model,
    device,
    class_weights,          # Tensor[num_classes]
    common_dim,
    model_name,
    text_model_encoder,
    attention_mecanism,
    results_folder_path
):
    # ----------------------------
    # Loss
    # ----------------------------
    if USE_FOCAL_LOSS:
        alpha = class_weights.clone().to(device)
        if ALPHA_SQRT:
            alpha = torch.sqrt(alpha)
        alpha = torch.clamp(alpha, ALPHA_CLAMP_MIN, ALPHA_CLAMP_MAX)
        criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, reduction="mean")
        criterion_name = f"focal(alpha={'sqrt' if ALPHA_SQRT else 'raw'} clamp[{ALPHA_CLAMP_MIN},{ALPHA_CLAMP_MAX}], gamma={FOCAL_GAMMA})"
    else:
        if USE_CLASS_WEIGHTS_IN_LOSS:
            w = torch.clamp(class_weights.to(device), 0.25, 10.0)
            criterion = nn.CrossEntropyLoss(weight=w)
            criterion_name = "cross_entropy(weighted_clamped)"
        else:
            criterion = nn.CrossEntropyLoss()
            criterion_name = "cross_entropy"

    # ----------------------------
    # Optimizer + Scheduler
    # ----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # Scheduler aligned with Balanced Accuracy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        verbose=True
    )

    model.to(device)

    # ----------------------------
    # Save paths
    # ----------------------------
    model_save_path = os.path.join(
        results_folder_path,
        f"model_{model_name}_with_{text_model_encoder}_{common_dim}_with_best_architecture"
    )
    os.makedirs(model_save_path, exist_ok=True)

    best_dir = os.path.join(model_save_path, f"{model_name}_fold_{fold_num}", "best-model")
    os.makedirs(best_dir, exist_ok=True)

    # Your EarlyStopping (kept as in your project)
    from utils.early_stopping import EarlyStopping
    early_stopping = EarlyStopping(
        patience=10,
        delta=0.005,
        verbose=True,
        path=str(best_dir) + "/",
        save_to_disk=True,
        early_stopping_metric_name="val_loss"
    )

    # ----------------------------
    # MLflow
    # ----------------------------
    experiment_name = "EXPERIMENTOS-MILK10K - MULTIMODAL (Sampler/Focal) - 2025-12-19"
    mlflow.set_experiment(experiment_name)

    train_losses, val_losses = [], []
    initial_time = time.time()

    with mlflow.start_run(
        run_name=f"{model_name}_{attention_mecanism}_fold_{fold_num}_heads_{num_heads}"
    ):
        mlflow.log_param("fold_num", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mecanism", attention_mecanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", criterion_name)
        mlflow.log_param("use_weighted_sampler", USE_WEIGHTED_SAMPLER)
        mlflow.log_param("use_focal_loss", USE_FOCAL_LOSS)
        mlflow.log_param("num_heads", num_heads)

        for epoch in range(num_epochs):
            # ------------------------
            # Train
            # ------------------------
            model.train()
            running_loss = 0.0

            for _, image, metadata, label in tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch+1}/{num_epochs}", leave=False):
                image = image.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / max(len(train_loader), 1)
            train_losses.append(float(train_loss))

            # ------------------------
            # Validate
            # ------------------------
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for _, image, metadata, label in val_loader:
                    image = image.to(device, non_blocking=True)
                    metadata = metadata.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)

                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_running_loss += loss.item()

            val_loss = val_running_loss / max(len(val_loader), 1)
            val_losses.append(float(val_loss))

            # ------------------------
            # Metrics
            # ------------------------
            metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                fold_num=fold_num,
                targets=targets,
                base_dir=model_save_path,
                model_name=model_name
            )

            metrics["epoch"] = int(epoch)
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)

            bacc = float(metrics.get("balanced_accuracy", 0.0))
            scheduler.step(bacc)

            current_lr = [pg["lr"] for pg in optimizer.param_groups]

            print(
                f"\n[Fold {fold_num}] Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} bacc={bacc:.4f} lr={current_lr}"
            )
            print(f"Metrics: {metrics}")

            # MLflow log
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating)):
                    mlflow.log_metric(k, float(v), step=epoch + 1)
                else:
                    mlflow.log_param(k, str(v))

            # Early stopping on bacc (aligned with scheduler objective)
            early_stopping(val_loss=val_loss, val_bacc=bacc, model=model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    # Load best weights and final eval
    train_process_time = time.time() - initial_time
    model = early_stopping.load_best_weights(model)
    model.eval()

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
    metrics["epochs"] = str(int(epoch))
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

    print(f"Model saved at {model_save_path}")
    return model, model_save_path


# ============================================================
# Pipeline (paper-safe)
# ============================================================

def pipeline(
    dataset_folder_path,
    image_type,
    num_epochs,
    batch_size,
    device,
    k_folds,
    common_dim,
    text_model_encoder,
    unfreeze_weights,
    attention_mecanism,
    model_name,
    num_heads,
    results_folder_path,
    num_workers=10,
    persistent_workers=True,
    type_of_problem:str = "multiclass"
):
    # Create two datasets: train has aug, val doesn't.
    # IMPORTANT: val will load the already-fitted OHE/scaler pickles (created by train).
    train_dataset = skinLesionDatasetsMILK10K.SkinLesionDataset(
        metadata_file=f"{dataset_folder_path}/MILK10k_Training_Metadata.csv",
        train_ground_truth=f"{dataset_folder_path}/MILK10k_Training_GroundTruth.csv",
        img_dir=f"{dataset_folder_path}/MILK10k_Training_Input",
        bert_model_name=text_model_encoder,
        image_encoder=model_name,
        drop_nan=False,
        random_undersampling=False,
        size=(224, 224),
        type_of_problem = type_of_problem,
        image_type=image_type,
        is_train=True
    )

    val_dataset = skinLesionDatasetsMILK10K.SkinLesionDataset(
        metadata_file=f"{dataset_folder_path}/MILK10k_Training_Metadata.csv",
        train_ground_truth=f"{dataset_folder_path}/MILK10k_Training_GroundTruth.csv",
        img_dir=f"{dataset_folder_path}/MILK10k_Training_Input",
        bert_model_name=text_model_encoder,
        image_encoder=model_name,
        drop_nan=False,
        random_undersampling=False,
        size=(224, 224),
        type_of_problem = type_of_problem,
        image_type=image_type,
        is_train=False
    )

    labels = np.array(train_dataset.labels, dtype=np.int64)
    targets = train_dataset.targets
    num_classes = len(targets)
    num_metadata_features = train_dataset.features.shape[1] if text_model_encoder == "one-hot-encoder" else 512

    print(f"Número de features do metadados: {num_metadata_features}\n")
    print(f"Classes presentes: {targets}\n")
    print(f"Número de classes: {num_classes}\n")

    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n==============================")
        print(f"Fold {fold}/{k_folds}")
        print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
        print(f"==============================\n")

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)

        # WeightedRandomSampler (Strategy A)
        sampler = None
        if USE_WEIGHTED_SAMPLER:
            fold_train_labels = labels[train_idx]
            counts = np.bincount(fold_train_labels, minlength=num_classes)
            weights_per_class = 1.0 / np.maximum(counts, 1)
            sample_weights = weights_per_class[fold_train_labels]

            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(fold_train_labels),
                replacement=True
            )

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True
        )

        # class weights (always safe length)
        fold_train_labels = labels[train_idx]
        class_weights = compute_class_weights(fold_train_labels, num_classes=num_classes).to(device)
        print(f"Pesos das classes (fold {fold}): {class_weights}")

        if attention_mecanism=="md-net":
            model = MDNet(
                meta_dim=num_metadata_features, 
                num_classes=num_classes, 
                unfreeze_weights=unfreeze_weights, 
                cnn_model_name=model_name,
                device=device
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
            results_folder_path=os.path.join(results_folder_path, str(num_heads), attention_mecanism)
        )

        save_predictions.model_val_predictions(
            model=model,
            dataloader=val_loader,
            device=device,
            fold_num=fold,
            targets=targets,
            base_dir=model_save_path,
            model_name=model_name
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    local_variables = load_local_variables.get_env_variables()

    num_epochs = int(local_variables["num_epochs"])
    batch_size = int(local_variables["batch_size"])
    k_folds = int(local_variables["k_folds"])
    common_dim = int(local_variables["common_dim"])
    num_workers = int(local_variables["num_workers"])
    list_num_heads = local_variables["list_num_heads"]

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
    type_of_problem="multiclass" # "binaryclass"
    image_type = "dermoscopic"  # or "clinical: close-up"
    
    attention_mecanism = "md-net"

    list_of_models = ["densenet169"]  # ["efficientnet-b0"] # ["mobilenet-v2", "davit_tiny.msft_in1k", "mvitv2_small.fb_in1k", "coat_lite_small.in1k", "caformer_b36.sail_in22k_ft_in1k", "vgg16", "densenet169", "resnet-50"]

    for model_name in list_of_models:
        for num_heads in list_num_heads:
            pipeline(
                dataset_folder_path=dataset_folder_path,
                image_type=image_type,
                num_epochs=num_epochs,
                batch_size=batch_size,
                device=device,
                k_folds=k_folds,
                common_dim=common_dim,
                text_model_encoder=text_model_encoder,
                unfreeze_weights=unfreeze_weights,
                attention_mecanism=attention_mecanism,
                model_name=model_name,
                num_heads=int(num_heads),
                results_folder_path=results_folder_path,
                num_workers=num_workers,
                persistent_workers=True,
                type_of_problem=type_of_problem
            )
