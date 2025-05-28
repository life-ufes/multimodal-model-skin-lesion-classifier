import torch
from torch import nn


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