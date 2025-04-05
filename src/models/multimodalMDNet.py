import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Corrected MetaNet module
class MetaNet(nn.Module):
    """
    Implementação da abordagem MetaNet
    Fusing Metadata and Dermoscopy Images for Skin Disease Diagnosis - https://ieeexplore.ieee.org/document/9098645
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(MetaNet, self).__init__()
        self.metanet = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 1),
            nn.ReLU(),
            nn.Conv2d(middle_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_maps, metadata):
        # metadata: tensor de dimensão [B, in_channels]
        # Adiciona dimensões espaciais para compatibilidade com Conv2d: [B, in_channels, 1, 1]
        metadata = metadata.unsqueeze(-1).unsqueeze(-1)
        # Passa os metadados pela sequência de convoluções para gerar um mapa de pesos
        x = self.metanet(metadata)
        # Multiplica o mapa de pesos com os mapas de features extraídos da imagem
        x = x * feat_maps
        return x

# Corrected MetaBlock module
class MetaBlock(nn.Module):
    """
    Implementação do Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U):
        """
        V: número de canais de features da imagem (ex.: 1664 da DenseNet-169)
        U: dimensão dos metadados (ex.: 85)
        """
        super(MetaBlock, self).__init__()
        self.fb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, img_features, metadata):
        # img_features: tensor de features da imagem com forma [B, V, H, W]
        # metadata: tensor de metadados com forma [B, U]
        t1 = self.fb(metadata)  # [B, V]
        t2 = self.gb(metadata)  # [B, V]
        # Expandir dimensões para compatibilidade com img_features (assumindo [B, V, H, W])
        t1 = t1.unsqueeze(-1).unsqueeze(-1)  # [B, V, 1, 1]
        t2 = t2.unsqueeze(-1).unsqueeze(-1)  # [B, V, 1, 1]
        # Modulação das features: aplica tanh na multiplicação e soma t2, seguido de sigmoid
        out = torch.sigmoid(torch.tanh(img_features * t1) + t2)
        return out

# Corrected MD-Net module
class MDNet(nn.Module):
    def __init__(self, meta_dim=85, num_classes=6, cnn_model_name="densenet169", text_model_name="one-hot-encode", hidden_dim=128, device="cpu", unfreeze_weights=False):
        super(MDNet, self).__init__()
        self.device = device
        self.num_channels = 1664  # Número de canais de saída da DenseNet-169
        self.meta_dim = meta_dim
        self.num_classes = num_classes
        self.cnn_model_name=cnn_model_name
        self.text_model_name = text_model_name
        # Carrega a DenseNet-169 pré-treinada e utiliza apenas o extrator de features
        densenet = models.densenet169(pretrained=True)
        # Controla se os pesos serão congelados ou não
        for param in densenet.parameters():
            param.requires_grad = unfreeze_weights
        self.feature_extractor = densenet.features
        
        # Módulo MetaNet: os metadados de dimensão meta_dim serão transformados em um mapa de pesos com num_channels
        self.meta_net = MetaNet(in_channels=meta_dim, middle_channels=hidden_dim, out_channels=self.num_channels)
        # Módulo MetaBlock: V = número de canais da imagem e U = dimensão dos metadados
        self.meta_block = MetaBlock(V=self.num_channels, U=meta_dim)
        
        # Classificador final
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.num_channels, self.num_classes)
    
    def forward(self, image, metadata):
        # image: [B, 3, H, W]
        # metadata: [B, meta_dim]
        # 1. Extrai as features visuais usando a DenseNet-169
        image_features = self.feature_extractor(image)  # [B, num_channels, H', W']
        
        # 2. Aplica o MetaNet para obter pesos canal a canal
        meta_net_features = self.meta_net(image_features, metadata)  # [B, num_channels, H', W']
        
        # 3. Aplica o MetaBlock para refinar as features com atenção orientada por metadados
        meta_block_features = self.meta_block(image_features, metadata)  # [B, num_channels, H', W']
        
        # 4. Fusão dos outputs (soma element-wise)
        fused_features = meta_net_features + meta_block_features
        
        # 5. Pooling global e classificação
        pooled_features = self.avg_pool(fused_features)  # [B, num_channels, 1, 1]
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # [B, num_channels]
        out = self.classifier(pooled_features)  # [B, num_classes]
        return out
