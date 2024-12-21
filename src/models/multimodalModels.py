import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

# Modelo Multimodal
class MultimodalModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(MultimodalModel, self).__init__()
        # CNN para imagens
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remover a camada final

        # Rede para metadados
        self.metadata_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Camada final combinada
        self.fc = nn.Sequential(
            nn.Linear(2048 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, image, metadata):
        image_features = self.cnn(image)
        metadata_features = self.metadata_fc(metadata)
        combined = torch.cat((image_features, metadata_features), dim=1)
        return self.fc(combined)

