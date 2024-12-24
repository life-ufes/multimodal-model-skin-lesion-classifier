import torch
import torch.nn as nn
from torchvision import models

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device, cnn_model_name, text_model_name, vocab_size=76):
        super(MultimodalModel, self).__init__()
        
        # Dimensões do modelo
        self.common_dim = 512
        self.text_encoder_dim_output = 512
        self.cnn_dim_output = 512

        # Modelo para imagens
        if cnn_model_name == "custom-cnn":
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(16 * 56 * 56, self.common_dim)
            )
        elif cnn_model_name == "resnet-50":
            self.image_encoder = models.resnet50(pretrained=True)
            self.cnn_dim_output = 2048
            # Congelar os pesos da ResNet-50
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Substituir a camada final por uma identidade
            self.image_encoder.fc = nn.Identity()
        elif cnn_model_name == "resnet-18":
            self.image_encoder = models.resnet18(pretrained=True)
            self.cnn_dim_output = 512
            # Congelar os pesos da ResNet-18
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Substituir a camada final por uma identidade
            self.image_encoder.fc = nn.Identity()

        elif cnn_model_name == "vgg16":
            self.image_encoder = models.vgg16(pretrained=True)
            self.cnn_dim_output = 4096
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Ajustar a saída para manter a dimensão esperada (4096)
            self.image_encoder.classifier = nn.Sequential(
                *list(self.image_encoder.classifier.children())[:-1],  # Remover a última camada (1000 classes)
                nn.Linear(4096, 4096)  # Garantir que a saída permanece 4096
            )

        elif cnn_model_name == "mobilenet-v2":
            self.image_encoder = models.mobilenet_v2(pretrained=True)
            self.cnn_dim_output = 1280
             # Congelar os pesos
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.fc = nn.Identity()
        else:
            raise ValueError("CNN não implementada.")
        
        # Projeção para os metadados (pré-processados)
        self.text_fc = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.text_encoder_dim_output)
        )

        # Projeções para o espaço comum
        self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)
        self.text_projector = nn.Linear(self.text_encoder_dim_output, self.common_dim)
        
        # Self-Attention Intra-Modality
        self.image_self_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        self.text_self_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        
        # Cross-Attention Inter-Modality
        self.image_cross_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        self.text_cross_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        
        # Camada de fusão
        self.fc_fusion = nn.Sequential(
            nn.Linear(self.common_dim * 2, 1024),
            nn.BatchNorm1d(1024), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, image, text_metadata):
        # Codificação de imagem
        image_features = self.image_encoder(image)
        image_features = self.image_projector(image_features)  # Projeção para dimensão comum
        image_features = image_features.unsqueeze(1)  # Shape: (batch_size, seq_length=1, common_dim)
        image_features = image_features.permute(1, 0, 2)  # Shape: (seq_length=1, batch_size, common_dim)

        # Projeção de metadados já pré-processados
        text_features = self.text_fc(text_metadata)  # Saída: (batch_size, common_dim)
        text_features = self.text_projector(text_features)
        text_features = text_features.unsqueeze(1)  # Shape: (batch_size, seq_length=1, common_dim)
        text_features = text_features.permute(1, 0, 2)  # Shape: (seq_length=1, batch_size, common_dim)
        
        # Self-Attention intra-modalidade
        image_features_att, _ = self.image_self_attention(image_features, image_features, image_features)
        text_features_att, _ = self.text_self_attention(text_features, text_features, text_features)
        
        # Cross-Attention inter-modalidade
        image_cross_att, _ = self.image_cross_attention(image_features_att, text_features_att, text_features_att)
        text_cross_att, _ = self.text_cross_attention(text_features_att, image_features_att, image_features_att)
        
        # Combinação das características
        combined_features = torch.cat([image_cross_att.squeeze(0), text_cross_att.squeeze(0)], dim=1)
        
        # Classificação final
        output = self.fc_fusion(combined_features)
        return output
