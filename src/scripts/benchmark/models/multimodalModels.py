import torch
import torch.nn as nn
import torchvision.models as models
from models import residualBlock

# Modelo Multimodal
class MultimodalModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(MultimodalModel, self).__init__()
        
        # CNN para imagens (ResNet50)
        self.cnn = models.resnet50(pretrained=True)
        
        # Congelar os pesos da ResNet50
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Substituir a camada final por uma identidade
        self.cnn.fc = nn.Identity()

        # Rede para os metadados
        self.metadata_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Camada combinada
        self.fc = nn.Sequential(
            nn.Linear(2048 + 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)# ,
            # nn.Softmax(dim=1)  # Softmax na saída
        )

    def forward(self, image, metadata):
        # Extrair features da CNN e da rede de metadados
        image_features = self.cnn(image)
        metadata_features = self.metadata_fc(metadata)
        
        # Combinar os dois vetores de features
        combined = torch.cat((image_features, metadata_features), dim=1)
        
        # Passar pela camada final
        return self.fc(combined)
