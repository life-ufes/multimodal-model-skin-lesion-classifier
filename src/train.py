import torch
import torch.nn as nn
from utils import model_metrics
from utils.early_stopping import EarlyStopping
import models.focalLoss as focalLoss
from models import multimodalIntraModal, multimodalModels, skinLesionDatasets, skinLesionDatasetsWithBert, multimodalEmbbeding, multimodalIntraInterModal
from utils.save_model_and_metrics import save_model_and_metrics
from collections import Counter
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import time
import os
import pandas as pd
from torch.utils.data import DataLoader, Subset
# Importações do MLflow
import mlflow

def compute_class_weights(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    return torch.tensor([class_weights[cls] for cls in sorted(class_counts.keys())], dtype=torch.float)

def train_process(num_epochs, fold_num, train_loader, val_loader, targets, model, device, weightes_per_category, model_name, text_model_encoder, attention_mecanism, results_folder_path):
    criterion = nn.CrossEntropyLoss(weight=weightes_per_category)
    # criterion = focalLoss.FocalLoss(alpha=None, gamma=2, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # ReduceLROnPlateau reduz o LR quando a métrica monitorada (val_loss) não melhora
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # Como estamos monitorando val_loss, queremos diminuir LR quando ela não melhora
        factor=0.1,       # Fator pelo qual a LR será multiplicada
        patience=2,       # Número de épocas sem melhoria antes de reduzir LR
        verbose=True      # Imprime quando há mudança de LR
    )
    model.to(device)

    # EarlyStopping
    early_stopping = EarlyStopping(patience=5, delta=0.01)
    # Registro do tempo de treinamento
    initial_time = time.time()
    # A época começa em zero
    epoch_index = 0
    # Iniciar uma execução no MLflow
    with mlflow.start_run(run_name=f"image_extractor__model_{model_name}_with_mecanism_{attention_mecanism}_fold_{fold_num}"):
        # Logar parâmetros no MLflow
        mlflow.log_param("fold_num", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", "cross_entropy")  # Ajuste conforme necessário

        for epoch_index in range(num_epochs):
            model.train()  # Ensure the model is in training mode
            running_loss = 0.0
            
            # Training loop
            for batch_index, (image, metadata, label) in enumerate(train_loader):
                image, metadata, label = image.to(device), metadata.to(device), label.to(device)

                optimizer.zero_grad()
                outputs = model(image, metadata)

                loss = criterion(outputs, label)
                loss.backward()
                
                optimizer.step()

                running_loss += loss.item()
            
            print(f"==="*40)
            # Average training loss for the epoch
            train_loss=running_loss/len(train_loader)
            print(f"\nTraining: Epoch {epoch_index}, Loss: {train_loss:.4f}")
            
            # Validation loop
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # No need to compute gradients during validation
                for image, metadata, label in val_loader:
                    image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                    
                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()

            # Calculate the average validation loss
            val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
            # === Atualiza o Scheduler de LR com base no val_loss ===
            scheduler.step(val_loss)
            
            # (Opcional) Verificar/Imprimir LR atual
            current_lr = [param_group['lr'] for param_group in optimizer.param_groups]
            print(f"Current Learning Rate(s): {current_lr}\n")
            # Evaluate metrics
            metrics, all_labels, all_predictions = model_metrics.evaluate_model(model, val_loader, device, fold_num)
            metrics["epoch"] = epoch_index
            
            metrics["train_loss"]=float(train_loss)
            metrics["val_loss"]=float(val_loss)
            print(f"Metrics: {metrics}")
            # Logar métricas no MLflow
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=epoch_index+1)
                else:
                    mlflow.log_param(metric_name, metric_value)  


            # Check early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    # Fim do treinamento
    train_process_time = time.time() - initial_time
    # Adição do tempo de treino nos registros
    metrics["train process time"]=str(train_process_time)
    metrics["epochs"]=str(int(epoch_index))
    metrics["data_val"]=str("val")
    # Salvar o modelo treinado
    model_save_path = os.path.join(results_folder_path, f"model_{model_name}_with_{text_model_encoder}_512")
    save_model_and_metrics(model, metrics, model_name, model_save_path, fold_num, all_labels, all_predictions, targets, data_val="val")
    print(f"Model saved at {model_save_path}")

    return model, model_save_path

def pipeline(dataset, num_metadata_features, num_epochs, batch_size, device, k_folds, num_classes, model_name, text_model_encoder, attention_mecanism, results_folder_path):        
    all_metrics = []
    # Criar o modelo e otimizador fora do loop do K-fold para manter os pesos
    model = multimodalIntraInterModal.MultimodalModel(num_classes, device, cnn_model_name=model_name, text_model_name=text_model_encoder, vocab_size=num_metadata_features, attention_mecanism=attention_mecanism)

    # Separar dados em treino, validação e teste
    test_size = int(0.2 * len(dataset))  # Usando 20% dos dados para teste
    indices = list(range(len(dataset)))
    train_val_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

    train_val_dataset = torch.utils.data.Subset(dataset, train_val_indices)  # Dados para treino e validação
    test_dataset = torch.utils.data.Subset(dataset, test_indices)  # Dados para teste final

    # Calcular pesos das classes para o treinamento com base nos dados de treino
    train_labels = [dataset.labels[i] for i in train_val_indices]
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"Pesos das classes a serem usadas: {class_weights}\n")
    # Obter os targets a serem usados
    targets = dataset.targets
    # Configuração do K-fold para os dados de treino e validação
    kFold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kFold.split(train_val_dataset)):
        print(f"Fold {fold+1}/{k_folds}")
        
        # Dividir os dados
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)
        
        # Criar os DataLoaders para treino e validação
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        # Treinar o modelo
        model, model_save_path = train_process(num_epochs, fold+1, train_loader, val_loader, targets, model, device, class_weights, model_name, text_model_encoder, attention_mecanism, results_folder_path)
        
        # Avaliação final no fold atual (com validação dentro do fold)
        metrics, all_labels, all_probabilities = model_metrics.evaluate_model(model, val_loader, device, fold+1)
        all_metrics.append(metrics)
        print(f"Metrics for fold {fold+1}: {metrics}")

    # Médias e desvios das métricas dos folds
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0]}

    print(f"Average Metrics (from folds): {avg_metrics}")
    print(f"Standard Deviation (from folds): {std_metrics}")

    # Validação final com o conjunto de teste
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Evaluate metrics
    final_metrics, all_labels, all_predictions = model_metrics.evaluate_model(model, test_loader, device, fold)
    # Adição do tempo de treino nos registros
    final_metrics["train process time"]=str(0)
    final_metrics["train_loss"]=str(0)
    final_metrics["val_loss"]=str(0)
    final_metrics["epochs"]=str(-1)
    final_metrics["data_val"]=str("test")
    final_metrics["fold"]=str("test")
    print(f"Final Test Metrics: {final_metrics}")
    save_model_and_metrics(model, final_metrics, model_name, model_save_path, -1, all_labels, all_predictions, dataset.targets, data_val="test")

def run_expirements(num_epochs, batch_size, k_folds, text_model_encoder, attention_mecanism, device):
    list_of_models= ["vit-base-patch16-224"] # ["vgg16", "mobilenet-v2", "resnet-18", "resnet-50", "vit-base-patch16-224"]
    
    for model_name in list_of_models:
        dataset = skinLesionDatasets.SkinLesionDataset(
        metadata_file="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/metadata.csv",
        img_dir="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/images",
        bert_model_name=text_model_encoder,
        image_encoder=model_name,
        drop_nan=False,
        random_undersampling=False
        )
        num_metadata_features = dataset.features.shape[1]
        print(f"Número de features do metadados: {num_metadata_features}\n")
        num_classes = len(dataset.metadata['diagnostic'].unique())

        pipeline(dataset, 
            num_metadata_features, 
            num_epochs, batch_size, 
            device, k_folds, num_classes, 
            model_name, text_model_encoder,
            attention_mecanism, 
            results_folder_path=f"/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/weights/{attention_mecanism}"
        )


if __name__ == "__main__":
    num_epochs = 100
    batch_size = 16
    k_folds=5 
    text_model_encoder= "one-hot-encoder" # 'one-hot-encoder'
    attention_mecanism="gated"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Treina todos modelos que podem ser usados no modelo multi-modal
    run_expirements(num_epochs, batch_size, k_folds, text_model_encoder, attention_mecanism, device)    