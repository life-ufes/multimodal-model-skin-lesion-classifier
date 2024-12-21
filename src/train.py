from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from utils import transforms, model_metrics
from models import multimodalModels, skinLesionDatasets
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
    for epoch_index in range(num_epochs):
        running_loss = 0.0
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
        
        print(f"Epoch {epoch_index}, Loss: {running_loss/100:.4f}")

        # Avaliar modelo após cada época
        metrics = model_metrics.evaluate_model(model, val_loader, device)

        # Formatar saída das métricas
        accuracy = metrics['accuracy']
        balanced_accuracy = metrics['balanced_accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        auc = metrics['auc']

        # Exibir métricas com tratamento para valores None
        print(
            f"Metrics - Accuracy: {accuracy:.4f}, "
            f"Balanced Accuracy: {balanced_accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"AUC: {auc:.4f}" if auc is not None else "AUC: Not Calculated"
        )

    return model

def pipeline(batch_size, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = skinLesionDatasets.SkinLesionDataset(
        metadata_file="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/metadata.csv",
        img_dir="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/images",
        transform=transforms.load_transforms(),
        drop_nan=False
    )
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    num_metadata_features = dataset.features.shape[1]
    num_classes = len(dataset.metadata['diagnostic'].unique())
    model = multimodalModels.MultimodalModel(num_metadata_features, num_classes)

    weightes_per_category = classweights_values(dataset.metadata['diagnostic'])

    # Obter os dados separados
    train_loader, val_loader = dataset.split_dataset(dataset, batch_size, test_size=0.3)
    trained_model = train_process(num_epochs, train_loader, val_loader, model, device, weightes_per_category.to(device))
    print("Training complete!")

if __name__ == "__main__":
    num_epochs = 10
    batch_size = 64
    pipeline(batch_size, num_epochs)
