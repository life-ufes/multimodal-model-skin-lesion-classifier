import torch
import torch.nn as nn
import os
import time
import gc
from collections import Counter

import numpy as np
import mlflow
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from models import skinLesionDatasetsWithPubMedEmbeddings
from models import multimodalintraintermodalemb as multimodalIntraInterModal
from utils import model_metrics, save_predictions
from utils.early_stopping import EarlyStopping
from utils import load_local_variables
from utils.save_model_and_metrics import save_model_and_metrics


# =====================================================
# Utilidades
# =====================================================
def compute_class_weights(labels, num_classes):
    """Calcula pesos por classe garantindo que toda classe tenha um peso definido."""
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


# =====================================================
# Treino / validação de um fold
# =====================================================
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
    results_folder_path,
    save_to_disk=False,
):
    criterion = nn.CrossEntropyLoss(weight=weightes_per_category)

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
        path=str(model_save_path + f"/{model_name}_fold_{fold_num}/best-model/"),
        save_to_disk=save_to_disk,
        early_stopping_metric_name="val_loss",
    )

    initial_time = time.time()
    epoch_index = 0
    train_losses = []
    val_losses = []

    experiment_name = "EXPERIMENTOS-PAD-UFES-20 - SENTENCE EMBEDDINGS (BERT/GPT2) - 2026-07-07"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=(
            f"image_extractor_model_{model_name}_with_mecanism_{attention_mecanism}"
            f"_text_{text_model_encoder}_fold_{fold_num}_num_heads_{num_heads}"
        )
    ):
        mlflow.log_param("fold_num", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mecanism", attention_mecanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", "weighted_cross_entropy")
        mlflow.log_param("num_heads", num_heads)

        for epoch_index in range(num_epochs):
            model.train()
            running_loss = 0.0

            for batch_index, (_, image, metadata, label) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch_index+1}/{num_epochs}", leave=False)
            ):
                image, label = image.to(device), label.to(device)

                # metadata pode ser um dict (BERT/GPT2: input_ids/attention_mask)
                if isinstance(metadata, dict):
                    metadata = {k: v.to(device) for k, v in metadata.items()}
                else:
                    metadata = metadata.to(device)

                optimizer.zero_grad()
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            print(f"\nTraining: Epoch {epoch_index}, Loss: {train_loss:.4f}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _, image, metadata, label in val_loader:
                    image, label = image.to(device), label.to(device)
                    if isinstance(metadata, dict):
                        metadata = {k: v.to(device) for k, v in metadata.items()}
                    else:
                        metadata = metadata.to(device)

                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            scheduler.step(val_loss)
            current_lr = [pg["lr"] for pg in optimizer.param_groups]
            print(f"Current Learning Rate(s): {current_lr}\n")

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
            print(f"Metrics: {metrics}")

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=epoch_index + 1)
                else:
                    mlflow.log_param(metric_name, metric_value)

            early_stopping(
                val_loss=val_loss, val_bacc=float(metrics["balanced_accuracy"]), model=model
            )
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))

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
        save_to_disk=save_to_disk,
        fold_num=fold_num,
        all_labels=all_labels,
        all_predictions=all_predictions,
        all_probabilities=all_probs,
        targets=targets,
        data_val="val",
        train_losses=train_losses,
        val_losses=val_losses,
    )
    print(f"Model saved at {model_save_path}")

    return model, model_save_path


# =====================================================
# Pipeline patient-wise (mesma estrutura do script principal)
# =====================================================
def pipeline(
    dataset,
    num_metadata_features: int,
    num_epochs: int,
    batch_size: int,
    device: str,
    k_folds: int,
    num_classes: int,
    model_name: str,
    num_heads: int,
    common_dim: int,
    text_model_encoder: str,
    unfreeze_weights,
    attention_mecanism: str,
    results_folder_path: str,
    num_workers: int = 10,
    persistent_workers: bool = False,  # mantido False: evita deadlock intermitente entre folds
    save_to_disk: bool = False,
):
    labels = [dataset.labels[i] for i in range(len(dataset))]
    groups = dataset.metadata["patient_id"].values  # split por paciente
    stratifiedKFold = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(
        stratifiedKFold.split(range(len(dataset)), labels, groups=groups)
    ):
        print(f"Fold {fold+1}/{k_folds}")

        train_labels = [labels[i] for i in train_idx]
        train_counts = Counter(train_labels)
        val_labels = [labels[i] for i in val_idx]
        val_counts = Counter(val_labels)
        print(f"Fold {fold+1}: train={train_counts}, val={val_counts}")

        if len(val_counts) < 2:
            print(f"⚠️ Fold {fold+1} skipped: validation set has only {len(val_counts)} class(es).")
            continue
        if len(train_counts) < 2:
            print(f"⚠️ Fold {fold+1} skipped: training set has only {len(train_counts)} class(es).")
            continue

        MIN_SAMPLES_PER_CLASS = 5
        if min(val_counts.values()) < MIN_SAMPLES_PER_CLASS:
            print(f"⚠️ Fold {fold+1} skipped: validation has less than {MIN_SAMPLES_PER_CLASS} samples in some class.")
            continue
        if min(train_counts.values()) < MIN_SAMPLES_PER_CLASS:
            print(f"⚠️ Fold {fold+1} skipped: training has less than {MIN_SAMPLES_PER_CLASS} samples in some class.")
            continue

        # BERT / GPT2 → Subset direto (tokenização já feita no __getitem__ do dataset)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        class_weights = compute_class_weights(train_labels, num_classes).to(device)
        print(f"Pesos das classes no fold {fold+1}: {class_weights}")

        sample_weights = torch.tensor(
            [class_weights[y].item() for y in train_labels], dtype=torch.float
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        # Sentence embeddings substituindo o one-hot como única entrada de texto:
        # o próprio MultimodalModel já roteia para o branch BERT/GPT2 quando
        # text_model_name != "one-hot-encoder" (ver multimodalIntraInterModal.py)
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
            n=1 if attention_mecanism == "no-metadata" else 2,
        )

        model, model_save_path = train_process(
            num_epochs=num_epochs,
            num_heads=num_heads,
            fold_num=fold + 1,
            train_loader=train_loader,
            val_loader=val_loader,
            targets=dataset.targets,
            model=model,
            device=device,
            weightes_per_category=class_weights,
            common_dim=common_dim,
            model_name=model_name,
            text_model_encoder=text_model_encoder,
            attention_mecanism=attention_mecanism,
            results_folder_path=results_folder_path,
            save_to_disk=save_to_disk,
        )

        save_predictions.model_val_predictions(
            model=model,
            dataloader=val_loader,
            device=device,
            fold_num=fold + 1,
            targets=dataset.targets,
            base_dir=model_save_path,
            model_name=model_name,
        )

        # Evita deadlock intermitente ao recriar DataLoaders no próximo fold
        del train_loader, val_loader
        gc.collect()


# =====================================================
# Loop de experimentos
# =====================================================
def run_expirements(
    dataset_folder_path: str,
    results_folder_path: str,
    llm_model_name_sequence_generator: str,
    llm_model_name: str,
    vllm_model_name: str,
    num_workers: int,
    num_epochs: int,
    batch_size: int,
    k_folds: int,
    common_dim: int,
    text_model_encoder: str,
    unfreeze_weights: str,
    device,
    list_num_heads: list,
    list_of_attention_mecanism: list,
    list_of_models: list,
    type_of_problem: str,
    save_to_disk: bool = False,
):
    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    dataset = skinLesionDatasetsWithPubMedEmbeddings.SkinLesionDataset(
                        metadata_file=(
                            f"{dataset_folder_path}/metadata_with_sentences_new-prompt-"
                            f"{llm_model_name_sequence_generator}.csv"
                        ),
                        img_dir=f"{dataset_folder_path}/images",
                        bert_model_name=text_model_encoder,
                        image_encoder=model_name,
                        drop_nan=False,
                    )

                    # Embeddings de sentença: dimensão fixa do encoder (768 para BERT/GPT2-base)
                    num_metadata_features = 512
                    if type_of_problem == "binaryclass":
                        num_classes = len(dataset.metadata["benign_malignant"].unique())
                    else:
                        num_classes = len(dataset.metadata["diagnostic"].unique())

                    pipeline(
                        dataset,
                        num_metadata_features=num_metadata_features,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        device=device,
                        k_folds=k_folds,
                        num_classes=num_classes,
                        model_name=model_name,
                        common_dim=common_dim,
                        text_model_encoder=text_model_encoder,
                        num_heads=num_heads,
                        unfreeze_weights=unfreeze_weights,
                        attention_mecanism=attention_mecanism,
                        results_folder_path=(
                            f"{results_folder_path}/{num_heads}/{attention_mecanism}"
                        ),
                        num_workers=num_workers,
                        persistent_workers=False,
                        save_to_disk=save_to_disk,
                    )
                except Exception as e:
                    print(
                        f"Erro ao processar o treino do modelo {model_name} "
                        f"e com o mecanismo: {attention_mecanism}. Erro:{e}\n"
                    )
                    continue


if __name__ == "__main__":
    local_variables = load_local_variables.get_env_variables()
    num_epochs = local_variables["num_epochs"]
    batch_size = local_variables["batch_size"]
    k_folds = local_variables["k_folds"]
    common_dim = local_variables["common_dim"]
    list_num_heads = local_variables["list_num_heads"]
    num_workers = int(local_variables["num_workers"])
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = local_variables["dataset_folder_path"]
    status_weights = str(local_variables["unfreeze_weights"])
    llm_model_name_sequence_generator = local_variables["LLM_MODEL_NAME_SEQUENCE_GENERATOR"]
    results_folder_path = str(local_variables["results_folder_path"])
    save_to_disk = bool(local_variables["save_to_disk"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    type_of_problem = "multiclass"

    # Sentence embeddings de verdade (fine-tunado com objetivo contrastivo),
    # substituindo o one-hot encoding. Opções: 100K/500K/1M/2M (checkpoints do
    # fine-tuning). Requer mean pooling no forward() do modelo (ver nota no chat).
    text_model_encoder = "pubmedbert-base-embeddings-500K"
    llm_model_name = "gemma3:27b"  # LLM que gerou as descrições textuais da imagem
    vllm_model_name = "qwen2.5:72b"

    results_folder_path = (
        f"{results_folder_path}/{dataset_folder_name}/{type_of_problem}/"
        f"sentence-embeddings-{text_model_encoder}/llm-{llm_model_name}/{status_weights}"
    )

    # Mesmos mecanismos de atenção comparados no experimento principal (one-hot),
    # agora com sentence embeddings como entrada de metadados
    list_of_attention_mecanism = [
        "att-intramodal+residual+cross-attention-metadados",  # modelo publicado (âncora)
        "rg-att-cross-modal",                                  # fiel à Fig. 1 da conferência
    ]
    list_of_models = ["caformer_b36.sail_in22k_ft_in1k"]

    run_expirements(
        dataset_folder_path=dataset_folder_path,
        results_folder_path=results_folder_path,
        llm_model_name_sequence_generator=llm_model_name_sequence_generator,
        llm_model_name=llm_model_name,
        vllm_model_name=vllm_model_name,
        num_workers=num_workers,
        num_epochs=num_epochs,
        batch_size=batch_size,
        k_folds=k_folds,
        common_dim=common_dim,
        text_model_encoder=text_model_encoder,
        unfreeze_weights=status_weights,
        device=device,
        list_num_heads=list_num_heads,
        list_of_attention_mecanism=list_of_attention_mecanism,
        list_of_models=list_of_models,
        type_of_problem=type_of_problem,
        save_to_disk=save_to_disk,
    )