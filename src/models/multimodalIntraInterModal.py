import torch
import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device, cnn_model_name, text_model_name, vocab_size=76):
        super(MultimodalModel, self).__init__()
        
        # Dimensões do modelo
        self.common_dim = 512
        
        # Modelo para imagens
        if cnn_model_name == "custom-cnn":
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(16 * 56 * 56, self.common_dim)
            )
        else:
            raise ValueError("CNN não implementada.")
        
        # Projeção para os metadados (pré-processados)
        self.text_fc = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.common_dim)
        )
        
        # Self-Attention Intra-Modality
        self.image_self_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        self.text_self_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        
        # Camada de fusão
        self.fc_fusion = nn.Sequential(
            nn.Linear(self.common_dim * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, image, text_metadata):
        # Codificação de imagem
        image_features = self.image_encoder(image)  # Saída: (batch_size, common_dim)
        image_features = image_features.unsqueeze(1)  # Shape: (batch_size, seq_length=1, common_dim)
        image_features = image_features.permute(1, 0, 2)  # Shape: (seq_length, batch_size, common_dim)

        # Projeção de metadados
        text_features = self.text_fc(text_metadata)  # Saída: (batch_size, common_dim)
        text_features = text_features.unsqueeze(1)  # Shape: (batch_size, seq_length=1, common_dim)
        text_features = text_features.permute(1, 0, 2)  # Shape: (seq_length, batch_size, common_dim)
        
        # Self-Attention intra-modalidade
        image_features_att, _ = self.image_self_attention(image_features, image_features, image_features)
        text_features_att, _ = self.text_self_attention(text_features, text_features, text_features)
        
        # Combinação das características
        combined_features = torch.cat([image_features_att.squeeze(0), text_features_att.squeeze(0)], dim=1)
        
        # Camada final de fusão e classificação
        output = self.fc_fusion(combined_features)
        return output
