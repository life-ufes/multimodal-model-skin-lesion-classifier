import torch
import torch.nn as nn
from utils import transforms, model_metrics
from utils.early_stopping import EarlyStopping
from models import multimodalIntraModal, multimodalModels, skinLesionDatasets, skinLesionDatasetsWithBert, multimodalEmbbeding, multimodalIntraInterModal
from utils.save_model_and_metrics import save_model_and_metrics
from collections import Counter
from sklearn.model_selection import KFold
import numpy as np
import time
import os
from torch.utils.data import DataLoader, Subset

def classweights_values(diagnostic_column):
    # Verificar se há valores NaN e remover (se necessário)
    diagnostic_column = diagnostic_column.dropna()

    # Garantir que 'diagnostic_column' tenha valores numéricos (se necessário)
    diagnostic_column = diagnostic_column.astype('category').cat.codes  # Convertendo para códigos de categoria inteiros

    # Obter os rótulos como uma lista
    all_labels = diagnostic_column.values

    # Verificar os valores únicos após conversão para inteiros
    print("Valores únicos após conversão para inteiros:", set(all_labels))

    # Contar as ocorrências de cada classe
    class_counts = Counter(all_labels)

    # Número total de amostras
    total_samples = len(all_labels)

    # Calcular pesos das classes: inverso da frequência
    class_weights = {}
    for class_idx, count in class_counts.items():
        # Garantir que não dividimos por 0
        weight = total_samples / (len(class_counts) * max(count, 1))  # 'max(count, 1)' previne divisão por zero
        class_weights[class_idx] = weight

    # Converter para tensor
    num_classes = len(class_counts)
    weights = torch.tensor([class_weights.get(i, 0.0) for i in range(num_classes)], dtype=torch.float)

    print("Pesos das classes:", class_weights)  # Verificar os pesos calculados
    
    return weights

def train_process(num_epochs, fold_num, train_loader, val_loader, model, device, weightes_per_category, model_name, text_model_encoder, results_folder_path):
    criterion = nn.CrossEntropyLoss(weight=weightes_per_category)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    model.to(device)

    # EarlyStopping
    early_stopping = EarlyStopping(patience=3, delta=0.01)
    # Registro do tempo de treinamento
    initial_time = time.time()
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
        
        print(f"==="*30)
        # Average training loss for the epoch
        print(f"\nTraining: Epoch {epoch_index}, Loss: {running_loss/len(train_loader):.4f}")
        
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
        
        # Evaluate metrics
        metrics, all_labels, all_predictions = model_metrics.evaluate_model(model, val_loader, device, fold_num)
        print(f"Metrics: {metrics}")

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # Fim do treinamento
    train_process_time = time.time() - initial_time
    # Adição do tempo de treino nos registros
    metrics["train process time"]=str(train_process_time)
    metrics["train_loss"]=str(float(running_loss/len(train_loader)))
    metrics["val_loss"]=str(float(val_loss))
    metrics["epochs"]=str(int(epoch_index))
    metrics["data_val"]=str("val")
    # Salvar o modelo treinado
    model_save_path = os.path.join(results_folder_path, f"model_{model_name}_with_{text_model_encoder}_512")
    save_model_and_metrics(model, metrics, model_name, model_save_path, fold_num, all_labels, all_predictions, dataset.targets, data_val="val")
    print(f"Model saved at {model_save_path}")

    return model, model_save_path

def pipeline(dataset, num_epochs, batch_size, device, k_folds, num_classes, model_name, text_model_encoder, results_folder_path):        
    # Criar o modelo e otimizador fora do loop do K-fold para manter os pesos
    model = multimodalIntraInterModal.MultimodalModel(num_classes, device, cnn_model_name=model_name, text_model_name=text_model_encoder)
    
    # Calcular pesos das classes para o treinamento
    class_weights = classweights_values(dataset.metadata['diagnostic']).to(device)
    
    # Configuração do K-fold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    all_metrics = []

    # Separar dados em treino, validação e teste
    test_size = int(0.2 * len(dataset))  # Usando 20% dos dados para teste
    train_val_dataset = torch.utils.data.Subset(dataset, range(test_size, len(dataset)))  # Dados para treino e validação
    test_dataset = torch.utils.data.Subset(dataset, range(test_size))  # Dados para teste final
    
    # Configuração do K-fold para os dados de treino e validação
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_dataset)):
        print(f"Fold {fold+1}/{k_folds}")
        
        # Dividir os dados
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)
        
        # Criar os DataLoaders para treino e validação
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Treinar o modelo
        model, model_save_path = train_process(num_epochs, fold+1, train_loader, val_loader, model, device, class_weights, model_name, text_model_encoder, results_folder_path)
        
        # Avaliação final no fold atual (com validação dentro do fold)
        metrics = model_metrics.evaluate_model(model, val_loader, device, fold+1)
        all_metrics.append(metrics)
        print(f"Metrics for fold {fold+1}: {metrics}")

    # Médias e desvios das métricas dos folds
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0]}

    print(f"Average Metrics (from folds): {avg_metrics}")
    print(f"Standard Deviation (from folds): {std_metrics}")

    # Validação final com o conjunto de teste
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    final_metrics = model_metrics.evaluate_model(model, test_loader, device, 'Final Test')
    # Evaluate metrics
    metrics, all_labels, all_predictions = model_metrics.evaluate_model(model, test_loader, device, fold)
    metrics["data_val"]=str("test")
    print(f"Final Test Metrics: {final_metrics}")
    save_model_and_metrics(model, metrics, model_name, model_save_path, fold, all_labels, all_predictions, dataset.targets, data_val="test")


if __name__ == "__main__":
    num_epochs = 25
    batch_size = 64
    k_folds=5 
    model_name="resnet-18"
    text_model_encoder= "albert-base-v2" # 'one-hot-encoder'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = skinLesionDatasetsWithBert.SkinLesionDataset(
        metadata_file="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/metadata.csv",
        img_dir="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/images",
        bert_model_name=text_model_encoder,
        image_encoder=model_name,
        drop_nan=False
    )
    num_metadata_features = dataset.metadata.shape[1]
    print(f"Número de features do metadados: {num_metadata_features}\n")
    num_classes = len(dataset.metadata['diagnostic'].unique())

    pipeline(dataset, num_epochs, batch_size, device, k_folds, num_classes, model_name, text_model_encoder, results_folder_path="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/weights")
