import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalModel(nn.Module):
    def __init__(self, cnn_model_name, num_classes, attention_heads=16, vocab_size=76):
        super(MultimodalModel, self).__init__()
        self.cnn_model_name = cnn_model_name
        self.cnn_output_dim = 2048
        self.text_encoder_output_dim = 64
        self.attention_heads = 8

        if self.cnn_model_name == "resnet-50":        
            self.cnn = models.resnet50(pretrained=True)
            self.cnn_dim_output = 2048

        elif self.cnn_model_name == "resnet-18":
            self.cnn = models.resnet18(pretrained=True)
            self.cnn_dim_output = 512
        
        # Congelar os pesos da ResNet50
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Substituir a camada final por uma identidade
        self.cnn.fc = nn.Identity()

        # Rede para os metadados
        self.metadata_fc = nn.Sequential(
            nn.Linear(vocab_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Camada de Atenção
        self.attention = nn.MultiheadAttention(embed_dim=(self.cnn_output_dim+ self.text_encoder_output_dim), num_heads=self.attention_heads, dropout=0.3)

        # Camada combinada
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_output_dim+self.text_encoder_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)  # Softmax na saída
        )

    def forward(self, image, metadata):
        # Extrair features da CNN e da rede de metadados
        image_features = self.cnn(image)  # (batch_size, 2048)
        metadata_features = self.metadata_fc(metadata)  # (batch_size, 64)

        # Concatenar as características de imagem e metadados
        combined_features = torch.cat((image_features, metadata_features), dim=1)  # (batch_size, 2112)

        # Ajuste de dimensões para a camada de atenção
        combined_features = combined_features.unsqueeze(0)  # (1, batch_size, 2112)
        
        # Aplicar atenção (Nota: para a MultiheadAttention, a entrada precisa ser (seq_len, batch_size, embed_dim))
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)  # (1, batch_size, 2112)
        
        # Remover a dimensão adicional
        attn_output = attn_output.squeeze(0)

        # Passar pelas camadas finais
        return self.fc(attn_output)
