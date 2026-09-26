import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loadImageModelClassifier import loadModels
from gatedResidualBlock import GatedAlteredResidualBlock


class RGDermNetSE(nn.Module):
    def __init__(self, num_classes, cnn_model_name, text_dim, common_dim=512,
                 backbone_train_mode="frozen_weights", dropout=0.3):
        super().__init__()
        self.image_encoder, cnn_dim = loadModels.loadModelImageEncoder(
            cnn_model_name, common_dim, backbone_train_mode=backbone_train_mode)
        self.image_projector = self._projector(cnn_dim, common_dim)
        self.text_projector = self._projector(text_dim, common_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(2, 1, common_dim))
        self.fusion = GatedAlteredResidualBlock(dim=common_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * common_dim, common_dim),
            nn.BatchNorm1d(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim // 2),
            nn.BatchNorm1d(common_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim // 2, num_classes),
        )

    @staticmethod
    def _projector(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image, metadata):
        img = self.image_encoder(image)
        if img.dim() == 4:
            img = img.mean(dim=(-2, -1))
        seq = torch.stack([self.image_projector(img),
                           self.text_projector(metadata.float())]) + self.modality_embedding
        fused = self.fusion(seq, seq, seq)
        return self.classifier(torch.cat([fused[0], fused[1]], dim=1))
