import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TabTransformer(nn.Module):
    def __init__(self, categorical_cardinalities, num_continuous, embed_dim=32, num_heads=4, num_transformer_layers=2, hidden_dim=128, output_dim=1, dropout=0.3):
        super(TabTransformer, self).__init__()

        # Embeddings para dados categóricos
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embed_dim) for num_categories in categorical_cardinalities
        ])
        
        self.num_categorical = len(categorical_cardinalities)
        self.embed_dim = embed_dim

        # Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            activation="relu", 
            dropout=dropout,  # Adicionando Dropout para regularização
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

        # Camada para dados numéricos (MLP simples)
        self.numeric_projection = nn.Linear(num_continuous, embed_dim) if num_continuous > 0 else None

        # Camada final de classificação
        self.fc = nn.Sequential(
            nn.Linear(self.num_categorical * embed_dim + (embed_dim if num_continuous > 0 else 0), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Adicionando Dropout na camada final
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_categorical, x_numerical):
        # Processar dados categóricos com embeddings
        categorical_embeds = [embed(x_categorical[:, i]) for i, embed in enumerate(self.embeddings)]
        categorical_embeds = torch.stack(categorical_embeds, dim=1)  # (batch_size, num_categorical, embed_dim)
        
        # Aplicar Transformer Encoder
        categorical_features = self.transformer_encoder(categorical_embeds)
        categorical_features = categorical_features.flatten(start_dim=1)  # Flatten (batch_size, num_categorical * embed_dim)

        # Processar dados numéricos (se houver)
        if self.numeric_projection:
            numerical_features = self.numeric_projection(x_numerical)  # (batch_size, embed_dim)
        else:
            numerical_features = torch.zeros((x_categorical.shape[0], 0), device=x_categorical.device)

        # Concatenar ambos
        features = torch.cat([categorical_features, numerical_features], dim=1)

        # Passar pela MLP final
        output = self.fc(features)
        return output
