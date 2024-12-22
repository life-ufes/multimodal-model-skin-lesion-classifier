from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from utils import transforms, model_metrics
from utils.early_stopping import EarlyStopping
from models import multimodalModels, skinLesionDatasets, multimodalEmbbeding
from collections import Counter

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

def train_process(num_epochs, train_loader, val_loader, model, device, weightes_per_categorie):
    criterion = nn.CrossEntropyLoss(weight=weightes_per_categorie)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    # EarlyStopping
    early_stopping = EarlyStopping(patience=5, delta=0.01)

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
            
            if batch_index % 100 == 99:
                last_loss = running_loss / 100
                print(f"[Epoch {epoch_index + 1}, Batch {batch_index + 1}] Loss: {last_loss:.4f}")
                running_loss = 0.0
        
        print(f"==="*30)
        # Average training loss for the epoch
        print(f"\nTraining: Epoch {epoch_index + 1}, Loss: {running_loss/len(train_loader):.4f}")
        
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
        metrics = model_metrics.evaluate_model(model, val_loader, device)

        # Extract metrics and display
        accuracy = metrics['accuracy']
        balanced_accuracy = metrics['balanced_accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        auc = metrics['auc']

        print(
            f"Metrics - Accuracy: {accuracy:.4f}, "
            f"Balanced Accuracy: {balanced_accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"AUC: {auc:.4f}" if auc is not None else "AUC: Not Calculated"
        )
        
        # Check early stopping condition
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            model.load_state_dict(early_stopping.get_best_model())  # Restore the best model weights
            break

    return model


def pipeline(batch_size, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = skinLesionDatasets.SkinLesionDataset(
        metadata_file="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/metadata.csv",
        img_dir="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/images",
        transform=transforms.load_transforms(),
        drop_nan=True
    )
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    num_metadata_features = dataset.features.shape[1]
    num_classes = len(dataset.metadata['diagnostic'].unique())
    model = multimodalEmbbeding.MultimodalModelWithEmbedding(num_metadata_features, num_classes)

    weightes_per_category = classweights_values(dataset.metadata['diagnostic'])

    # Obter os dados separados
    train_loader, val_loader = dataset.split_dataset(dataset, batch_size, test_size=0.3)
    trained_model = train_process(num_epochs, train_loader, val_loader, model, device, weightes_per_category.to(device))
    
    # Salvar o modelo
    torch.save(trained_model, "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/weights/multimodal_resnet50_embbeding.pth")
    print("Modelo salvo!")


if __name__ == "__main__":
    num_epochs = 100
    batch_size = 128
    pipeline(batch_size, num_epochs)
