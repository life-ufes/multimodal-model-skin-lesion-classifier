import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import transforms
from models import multimodalModels, skinLesionDatasets

def train_process(num_epochs, dataloader, model, device):
    # Configuração de treinamento
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)  # Enviar modelo para GPU, se disponível
    for epoch_index in range(num_epochs):
        running_loss = 0.0
        for batch_index, (image, metadata, label) in enumerate(dataloader):
            # Enviar dados para o dispositivo (CPU/GPU)
            image, metadata, label = image.to(device), metadata.to(device), label.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(image, metadata)

            # Compute the loss and its gradients
            loss = criterion(outputs, label)
            
            loss.backward()

            print(f"Loss {loss.item()}\n")
            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if batch_index % 100 == 99:
                last_loss = running_loss / 100  # Loss por lote
                print(f"[Epoch {epoch_index + 1}, Batch {batch_index + 1}] Loss: {last_loss:.4f}")
                running_loss = 0.0
    return model


def pipeline(batch_size, num_epochs):
    # Configuração do dispositivo (CPU ou GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar dataset
    dataset = skinLesionDatasets.SkinLesionDataset(
        metadata_file="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/metadata.csv",
        img_dir="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/images",
        transform=transforms.load_transforms()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Inicializar modelo
    num_metadata_features = dataset.features.shape[1]
    num_classes = len(dataset.metadata['diagnostic'].unique())
    model = multimodalModels.MultimodalModel(num_metadata_features, num_classes)


    # Inicia o processo de treino
    trained_model = train_process(num_epochs, dataloader, model, device)
    print("Treinamento concluído!")

if __name__ == "__main__":
    # Treino de 'n' épocas
    num_epochs = 10
    batch_size=32
    pipeline(batch_size, num_epochs)
