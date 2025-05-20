import torch
import torch.nn as nn
from utils import model_metrics
from utils.early_stopping import EarlyStopping
import models.focalLoss as focalLoss
from models import multimodalIntraModalWithBert, multimodalModels, skinLesionDatasets, skinLesionDatasetsPAD2025, skinLesionDatasetsWithBert, multimodalEmbbeding, multimodalIntraInterModal, multimodalIntraInterModalToOptimzeAfterFIneTunning
from utils.save_model_and_metrics import save_model_and_metrics
from collections import Counter
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import numpy as np
import time
import os
import pandas as pd
from torch.utils.data import DataLoader, Subset
# Importações do MLflow
import mlflow
from tqdm import tqdm


def compute_class_weights(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    return torch.tensor([class_weights[cls] for cls in sorted(class_counts.keys())], dtype=torch.float)

def train_process(num_epochs, 
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
                  results_folder_path):

    criterion = nn.CrossEntropyLoss(weight=weightes_per_category)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        verbose=True
    )
    model.to(device)

    # Save the final (or best) model
    model_save_path = os.path.join(
        results_folder_path, 
        f"model_{model_name}_with_{text_model_encoder}_{common_dim}_with_best_architecture"
    )

    os.makedirs(model_save_path, exist_ok=True)
    print(model_save_path)

    # Instantiate EarlyStopping
    # Make sure EarlyStopping stores model.state_dict(), not the entire model.
    early_stopping = EarlyStopping(
        patience=5, 
        delta=0.01, 
        verbose=True,
        path=str(model_save_path+f'/{str(fold_num)}/best-model/'),   # Where to save the best weights (optional)
        save_to_disk=False       # If True, saves best weights to 'best_model.pt'
    )

    initial_time = time.time()
    epoch_index = 0  # Track the epoch

    # Set your MLflow experiment
    experiment_name = "EXPERIMENTOS-PAD-UFES25 - RESIDUAL BLOCK USAGE"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=(
            f"image_extractor_model_{model_name}_with_mecanism_"
            f"{attention_mecanism}_fold_{fold_num}_num_heads_{num_heads}"
        )
    ):
        # Log MLflow parameters
        mlflow.log_param("fold_num", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mecanism", attention_mecanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", "cross_entropy")
        mlflow.log_param("num_heads", num_heads)

        # -----------------------------
        # Training Loop
        # -----------------------------
        for epoch_index in range(num_epochs):
            model.train()
            running_loss = 0.0

            # Adicionando barra de progresso para o loop de batches
            for batch_index, (image, metadata, label) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch_index+1}/{num_epochs}", leave=False)):
                image, metadata, label = (
                    image.to(device),
                    metadata.to(device),
                    label.to(device)
                )

                optimizer.zero_grad()
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            print(f"\nTraining: Epoch {epoch_index}, Loss: {train_loss:.4f}")

            # -----------------------------
            # Validation Loop
            # -----------------------------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for image, metadata, label in val_loader:
                    image, metadata, label = (
                        image.to(device),
                        metadata.to(device),
                        label.to(device)
                    )
                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Step the scheduler with validation loss
            scheduler.step(val_loss)
            current_lr = [pg['lr'] for pg in optimizer.param_groups]
            print(f"Current Learning Rate(s): {current_lr}\n")

            # -----------------------------
            # Evaluate Metrics
            # -----------------------------
            metrics, all_labels, all_predictions = model_metrics.evaluate_model(
                model, val_loader, device, fold_num
            )
            metrics["epoch"] = epoch_index
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)
            print(f"Metrics: {metrics}")

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=epoch_index + 1)
                else:
                    mlflow.log_param(metric_name, metric_value)

            # -----------------------------
            # Early Stopping
            # -----------------------------
            early_stopping(val_loss, model)

            # Check if we should stop early
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    # Load the best model weights
    early_stopping.load_best_weights(model)

    # End of training
    train_process_time = time.time() - initial_time
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
        targets=targets, 
        data_val="val"
    )
    print(f"Model saved at {model_save_path}")

    return model, model_save_path



def pipeline(dataset, num_metadata_features, num_epochs, batch_size, device, k_folds, num_classes, model_name, num_heads, common_dim, text_model_encoder, unfreeze_weights, attention_mecanism, results_folder_path):
    all_metrics = []

    # Obter os rótulos para validação estratificada (se necessário)
    labels = [dataset.labels[i] for i in range(len(dataset))]

    # Configurar o K-Fold
    stratifiedKFold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(stratifiedKFold.split(range(len(dataset)), labels)):
        print(f"Fold {fold+1}/{k_folds}")

        # Criar datasets para treino e validação do fold atual
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Criar DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=15)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=15)

        # Calcular pesos das classes com base no conjunto de treino
        train_labels = [labels[i] for i in train_idx]
        class_weights = compute_class_weights(train_labels).to(device)
        print(f"Pesos das classes no fold {fold+1}: {class_weights}")

        # Criar o modelo
        model = multimodalIntraInterModal.MultimodalModel(num_classes, num_heads, device, cnn_model_name=model_name, text_model_name=text_model_encoder, common_dim=common_dim, vocab_size=num_metadata_features, unfreeze_weights=unfreeze_weights, attention_mecanism=attention_mecanism, n=1 if attention_mecanism=="no-metadata" else 2)
        # Treinar o modelo no fold atual
        model, model_save_path = train_process(
            num_epochs, num_heads, fold+1, train_loader, val_loader, dataset.targets, model, device,
            class_weights, common_dim, model_name, text_model_encoder, attention_mecanism, results_folder_path
        )


def run_expirements(dataset_folder_path:str, results_folder_path:str, num_epochs:int, batch_size:int, k_folds:int, common_dim:int, text_model_encoder:str, unfreeze_weights: bool, device, list_num_heads: list, list_of_attention_mecanism:list, list_of_models: list):
    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    if (text_model_encoder=='one-hot-encoder' or text_model_encoder=="tab-transformer"):
                        dataset = skinLesionDatasetsPAD2025.SkinLesionDataset(
                        metadata_file=f"{dataset_folder_path}/anonymous-metadata.csv",
                        img_dir=f"{dataset_folder_path}/anonymous-images",
                        bert_model_name=text_model_encoder,
                        image_encoder=model_name,
                        drop_nan=False)
                    else:
                        dataset = skinLesionDatasetsPAD2025.SkinLesionDataset(
                        metadata_file=f"{dataset_folder_path}/anonymous-metadata.csv",
                        img_dir=f"{dataset_folder_path}/anonymous-images",
                        bert_model_name=text_model_encoder,
                        image_encoder=model_name,
                        drop_nan=False)

                    num_metadata_features = dataset.features.shape[1]
                    print(f"Número de features do metadados: {num_metadata_features}\n")
                    num_classes = len(dataset.metadata['macroCIDDiagnostic'].unique())

                    pipeline(dataset, 
                        num_metadata_features=num_metadata_features, 
                        num_epochs=num_epochs, batch_size=batch_size, 
                        device=device, k_folds=k_folds, num_classes=num_classes, 
                        model_name=model_name, common_dim=common_dim, 
                        text_model_encoder=text_model_encoder,
                        num_heads=num_heads,
                        unfreeze_weights=unfreeze_weights,
                        attention_mecanism=attention_mecanism, 
                        results_folder_path=f"{results_folder_path}/{num_heads}/{attention_mecanism}"
                    )
                except Exception as e:
                    print(f"Erro ao processar o treino do modelo {model_name} e com o mecanismo: {attention_mecanism}. Erro:{e}\n")
                    continue

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 32
    k_folds=5
    common_dim=512
    text_model_encoder = 'one-hot-encoder' #  'bert-base-uncased' # 'one-hot-encoder' # 'tab-transformer'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    list_num_heads=[8]
    dataset_folder_name="PAD-UFES-25"
    dataset_folder_path=f"/data/{dataset_folder_name}"
    unfreeze_weights = True # Caso queira descongelar os pesos da CNN desejada
    results_folder_path = f"./src/results/testes/testes-da-implementacao-final/{dataset_folder_name}/{'unfrozen_weights' if unfreeze_weights else 'frozen_weights'}"
    # Para todas os tipos de estratégias a serem usadas
    list_of_attention_mecanism = ["att-intramodal+residual+cross-attention-metadados"] #  ["concatenation", "no-metadata", "att-intramodal+residual", "att-intramodal+residual+cross-attention-metadados", "att-intramodal+residual+cross-attention-metadados+att-intramodal+residual"] # ["weighted-after-crossattention", "cross-weights-after-crossattention", "crossattention", "concatenation", "no-metadata", "weighted"]
    # Testar com todos os modelos
    list_of_models = ["nextvit_small.bd_ssld_6m_in1k", "coat_lite_small.in1k", "caformer_b36.sail_in22k_ft_in1k", "beitv2_large_patch16_224.in1k_ft_in22k_in1k", "vgg16", "mobilenet-v2", "densenet169", "resnet-50"] # ["nextvit_small.bd_ssld_6m_in1k", "mvitv2_small.fb_in1k", "coat_lite_small.in1k", "davit_tiny.msft_in1k", "caformer_b36.sail_in22k_ft_in1k", "beitv2_large_patch16_224.in1k_ft_in22k_in1k", "vgg16", "mobilenet-v2", "densenet169", "resnet-50"]
    # Treina todos modelos que podem ser usados no modelo multi-modal
    run_expirements(dataset_folder_path, results_folder_path, num_epochs, batch_size, k_folds, common_dim, text_model_encoder, unfreeze_weights, device, list_num_heads, list_of_attention_mecanism=list_of_attention_mecanism, list_of_models=list_of_models)    
