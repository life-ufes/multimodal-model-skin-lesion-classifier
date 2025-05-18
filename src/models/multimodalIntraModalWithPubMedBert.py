import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loadImageModelClassifier import loadModels


class MultimodalModel(nn.Module):
    def __init__(self, num_classes, num_heads, device, cnn_model_name, text_model_name, common_dim=512, vocab_size=85, unfreeze_weights=False, attention_mecanism="combined", n=2):
        super(MultimodalModel, self).__init__()
        # Dimensões do modelo
        self.common_dim = common_dim
        self.text_encoder_dim_output = 512
        self.cnn_dim_output = 512
        self.device = device
        self.cnn_model_name = cnn_model_name
        self.text_model_name = text_model_name
        self.attention_mecanism = attention_mecanism
        self.num_heads = num_heads  # para MultiheadAttention
        self.n = n 
        self.num_classes = num_classes
        self.unfreeze_weights_of_visual_feat_extractor = unfreeze_weights
        # -------------------------
        # 1) Image Encoder
        # -------------------------
        self.image_encoder, self.cnn_dim_output = loadModels.loadModelImageEncoder(
            self.cnn_model_name,
            self.common_dim,
            unfreeze_weights=self.unfreeze_weights_of_visual_feat_extractor
        )
        # Projeção para o espaço comum da imagem (ex.: 512 -> self.common_dim)
        # self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)
        
        # Definir encoder de texto (BERT)
        # Carrega BERT, Bart, etc., congelado
        self.text_encoder, self.text_encoder_dim_output, vocab_size = loadModels.loadTextModelEncoder(
            text_model_encoder=self.text_model_name, unfreeze_weights=self.unfreeze_weights_of_visual_feat_extractor)
        # Projeta 768 (ou 1024) -> 512
        self.text_fc = nn.Sequential(
            nn.Linear(vocab_size, self.text_encoder_dim_output),
            nn.ReLU(),
            nn.Dropout(0.3))
        
        # Projeção final p/ espaço comum
        # self.text_projector = nn.Linear(self.text_encoder_dim_output, self.common_dim)
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder_dim_output, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, common_dim)
        )
        self.image_projector = nn.Sequential(
            nn.Linear(self.cnn_dim_output, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, common_dim)
        )

        # Camada de Fusão Final
        # -------------------------
        self.fc_fusion = self.fc_mlp_module(n=self.n)
    def fc_mlp_module(self, n=1):
        fc_fusion = nn.Sequential(
            nn.Linear(self.common_dim * n, self.common_dim),
            nn.BatchNorm1d(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.BatchNorm1d(self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim // 2, self.num_classes),
            nn.Softmax(dim=1)
        )
        return fc_fusion
        
    def forward(self, images, metadata):
            # Move inputs to device
            images = images.to(self.device)

            # Visual features
            vis_feat = self.image_encoder(images)
            # Ensure float32 dtype
            vis_feat = vis_feat.float()
            vis_proj = self.image_projector(vis_feat)

            # Textual features
            if isinstance(metadata, dict):
                input_ids = metadata['input_ids'].to(self.device)
                attention_mask = metadata['attention_mask'].to(self.device)
                text_out = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # BERT uses CLS token, GPT2 use last hidden state
                if 'gpt2' in self.text_model_name:
                    txt_feat = text_out.last_hidden_state[:, -1, :]
                else:
                    txt_feat = text_out.last_hidden_state[:, 0, :]
            else:
                # metadata is precomputed embedding tensor
                # Move and ensure float32
                txt_feat = metadata.to(self.device).float()

            txt_proj = self.text_projector(txt_feat)

            # Concatenate projected features
            combined = torch.cat([vis_proj, txt_proj], dim=1)

            # Classification
            logits = self.fc_fusion(combined)
            return logits
