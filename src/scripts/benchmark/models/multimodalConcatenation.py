import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformers import ViTFeatureExtractor, CLIPProcessor, CLIPModel

from loadImageModelClassifier import loadModels

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device, cnn_model_name, text_model_name, vocab_size=85, attention_mecanism="combined"):
        super(MultimodalModel, self).__init__()
        
        # Dimensões do modelo
        self.common_dim = 512
        self.text_encoder_dim_output = 512
        self.cnn_dim_output = 512
        self.device = device
        self.cnn_model_name = cnn_model_name
        self.text_model_name = text_model_name
        self.attention_mecanism = attention_mecanism
        self.num_heads = 8  # para MultiheadAttention

        # -------------------------
        # 1) Image Encoder
        # -------------------------
        self.image_encoder, self.cnn_dim_output = loadModels.loadModelImageEncoder(
            self.cnn_model_name,
            self.common_dim
        )
        
        # Se for ViT, teremos ViTFeatureExtractor
        if self.cnn_model_name == "vit-base-patch16-224":
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(
                f"google/{self.cnn_model_name}"
            )
        elif self.cnn_model_name == "openai/clip-vit-base-patch16":
            self.feature_extractor = CLIPProcessor.from_pretrained(self.cnn_model_name)
            self.image_encoder = CLIPModel.from_pretrained(self.cnn_model_name)

        # Projeção para o espaço comum da imagem (ex.: 512 -> self.common_dim)
        self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)

        # -------------------------
        # 2) Text Encoder
        # -------------------------
        if self.text_model_name == "one-hot-encoder":
            # Metadados / one-hot -> FC
            self.text_fc = nn.Sequential(
                nn.Linear(vocab_size, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, self.text_encoder_dim_output)
            )
        else:
            # Carrega BERT, Bart, etc., congelado
            self.text_encoder, self.text_encoder_dim_output = loadModels.loadTextModelEncoder(
                text_model_name
            )
            # Projeta 768 (ou 1024) -> 512
            self.text_fc = nn.Sequential(
                nn.Linear(768, self.text_encoder_dim_output),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        # Projeção final p/ espaço comum
        self.text_projector = nn.Linear(self.text_encoder_dim_output, self.common_dim)

        # -------------------------
        # 3) Atenções Intra e Inter
        # -------------------------
        self.image_self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=False
        )
        self.text_self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=False
        )
        
        self.image_cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=False
        )
        self.text_cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=False
        )

        # -------------------------
        # 4) Gating Mechanisms
        # -------------------------
        self.img_gate = nn.Linear(self.common_dim, self.common_dim)
        self.txt_gate = nn.Linear(self.common_dim, self.common_dim)

        # -------------------------
        # 5) Camada de Fusão Final
        # -------------------------
        self.fc_fusion = nn.Sequential(
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.BatchNorm1d(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.BatchNorm1d(self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim // 2, num_classes) #,
            # nn.Softmax(dim=1)
        )
    
    def forward(self, image, text_metadata):
        """
        image: tensor de imagens (batch, C, H, W) se CNN 
               ou lista PIL se ViTFeatureExtractor
        text_metadata: dicionário c/ 'input_ids', 'attention_mask' (BERT/Bart)
                       ou tensor se "one-hot-encoder".
        """

        # === [A] Extrator de Imagem ===
        if self.cnn_model_name == ("vit-base-patch16-224" or "openai/clip-vit-base-patch16"):
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            outputs = self.image_encoder(**inputs)
            # outputs.last_hidden_state => (batch, seq_len_img, hidden_dim)
            image_features = outputs.last_hidden_state
        else:
            # CNN -> (batch, cnn_dim_output)
            image_features = self.image_encoder(image)
            # Dá forma (batch, 1, cnn_dim_output)
            image_features = image_features.unsqueeze(1)

        # Projeção p/ espaço comum
        b_i, s_i, d_i = image_features.shape
        image_features = image_features.view(b_i*s_i, d_i)
        projected_image_features = self.image_projector(image_features)
        image_features = projected_image_features.view(b_i, s_i, -1)
        # -> (seq_len_img, batch, common_dim)
        image_features = image_features.permute(1, 0, 2)

        # === [B] Extrator de Texto ===
        if self.text_model_name == "one-hot-encoder":
            text_features = self.text_fc(text_metadata)  # (batch, 512)
            text_features = text_features.unsqueeze(1) # Adiciona uma dimensão às features
        else:
            # Ajustar input_ids e attention_mask p/ shape [batch, seq_len]
            input_ids = text_metadata["input_ids"]
            attention_mask = text_metadata["attention_mask"]

            if len(input_ids.shape) == 3:  # por ex. (batch, 1, seq_len)
                input_ids = input_ids.squeeze(1)
            if len(attention_mask.shape) == 3:
                attention_mask = attention_mask.squeeze(1)

            encoder_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_features = encoder_output.last_hidden_state  # (batch, seq_len_text, 768)

            b_t, s_t, d_t = text_features.shape
            text_features = text_features.view(b_t*s_t, d_t)
            text_features = self.text_fc(text_features)  # ex.: (batch*seq_t, 512)
            text_features = text_features.view(b_t, s_t, -1)

        # Projeção para espaço comum
        b_tt, s_tt, d_tt = text_features.shape
        text_features = text_features.view(b_tt*s_tt, d_tt)
        projected_text_features = self.text_projector(text_features)
        text_features = projected_text_features.view(b_tt, s_tt, -1)
        text_features = text_features.permute(1, 0, 2)

        # === [C] Self-Attention Intra-Modality ===
        image_features_att, _ = self.image_self_attention(
            image_features, image_features, image_features
        )
        text_features_att, _ = self.text_self_attention(
            text_features, text_features, text_features
        )

        # === [D] Cross-Attention Inter-Modality ===
        # "Imagem assiste ao texto"
        image_cross_att, _ = self.image_cross_attention(
            query=image_features_att,
            key=text_features_att,
            value=text_features_att
        )
        # "Texto assiste à imagem"
        text_cross_att, _ = self.text_cross_attention(
            query=text_features_att,
            key=image_features_att,
            value=image_features_att
        )

        # === [E] Pooling das atenções finais 
        image_cross_att = image_cross_att.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
        text_cross_att = text_cross_att.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

        image_pooled = image_cross_att.mean(dim=1)  # (batch, common_dim)
        text_pooled = text_cross_att.mean(dim=1)    # (batch, common_dim)

        if self.attention_mecanism == "weighted":
            # # === [F] Gating: quanto usar de 'peso' para cada modal
            if self.cnn_model_name=="vit-base-patch16-224":
                # Os modelos ViT possuem uma sequência de tokens que precisa ser processada antes de ser projetada
                projected_image_features = projected_image_features.view(b_i, s_i, -1).mean(dim=1)  # (batch, common_dim)
                projected_text_features = projected_text_features.view(b_tt, s_tt, -1).mean(dim=1)  # (batch, common_dim)

            alpha_img = torch.sigmoid(self.img_gate(projected_image_features))  # (batch, common_dim)
            alpha_txt = torch.sigmoid(self.txt_gate(projected_text_features))   # (batch, common_dim)
            
            # Multiplicamos as features pela máscara gerada
            image_pooled_gated = alpha_img * projected_image_features
            text_pooled_gated = alpha_txt * projected_text_features
            combined_features = torch.cat([image_pooled_gated, text_pooled_gated], dim=1)

        if self.attention_mecanism == "concatenation":
            # 
            if self.cnn_model_name=="vit-base-patch16-224":
                # Os modelos ViT possuem uma sequência de tokens que precisa ser processada antes de ser projetada
                projected_image_features = projected_image_features.view(b_i, s_i, -1).mean(dim=1)  # (batch, common_dim)
                projected_text_features = projected_text_features.view(b_tt, s_tt, -1).mean(dim=1)  # (batch, common_dim)
            
            # Apenas concatena as features projetadas
            combined_features = torch.cat([projected_image_features, projected_text_features], dim=1)

        else:
            raise KeyError(f"Este módulo processa as features de forma concatenada!\n")

        output = self.fc_fusion(combined_features)  # (batch, num_classes)
        return output
