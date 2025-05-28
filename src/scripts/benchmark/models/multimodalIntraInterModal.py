import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformers import ViTFeatureExtractor
# from residualBlock import ResidualBlock
# from residualBlockADeepBasedMultimodal import ResidualBlock
from gatedResidualBlock import GatedAlteredResidualBlock
from loadImageModelClassifier import loadModels
from metablock import MetaBlock

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

        # Se for ViT, teremos ViTFeatureExtractor
        if self.cnn_model_name in ["google/vit-base-patch16-224","openai/clip-vit-base-patch16", "facebookresearch/dinov2"]:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(
                f"{self.cnn_model_name}"
            )

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
            self.text_encoder, self.text_encoder_dim_output, vocab_size = loadModels.loadTextModelEncoder(
                text_model_encoder = text_model_name, unfreeze_weights = self.unfreeze_weights_of_visual_feat_extractor
            )
            # Projeta 768 (ou 1024) -> 512
            self.text_fc = nn.Sequential(
                nn.Linear(vocab_size, self.text_encoder_dim_output),
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
        self.fc_fusion = self.fc_mlp_module(n=self.n)
       # 6) Residual Blocks
        # -------------------------
        self.image_residual = GatedAlteredResidualBlock(dim=self.common_dim)
        self.text_residual = GatedAlteredResidualBlock(dim=self.common_dim)

        self.fc_no_mlp_to_visual_cls = nn.Sequential(
            nn.Linear(self.cnn_dim_output, self.num_classes),
            nn.Softmax(dim=1)
        )

        # Bloco do Metablock, caso queira usar
        self.meta_block = MetaBlock(V=self.cnn_dim_output, U=self.text_encoder_dim_output)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

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

    def forward(self, image, text_metadata):
        """
        image: tensor de imagens (batch, C, H, W) se CNN 
        ou lista PIL se ViTFeatureExtractor
        text_metadata: dicionário c/ 'input_ids', 'attention_mask' (BERT/Bart)
        ou tensor se "one-hot-encoder".
        """
        image = image.to(self.device)
        # === [A] Image Feature Extraction ===
        if self.cnn_model_name in ["google/vit-base-patch16-224", "openai/clip-vit-base-patch16"]:
            # Use the feature extractor (e.g., CLIPProcessor) to preprocess the image
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            
            # Move input tensors to the correct device
            inputs = {k: v for k, v in inputs.items()}
            
            # Forward pass through the vision encoder
            outputs = self.image_encoder(**inputs)
            
            # Extract feature representations
            image_features_before = outputs.last_hidden_state  # (batch, seq_len_img, hidden_dim)
        else:
            # CNN -> (batch, cnn_dim_output)
            image_features = self.image_encoder(image) #.to(self.device)
            # Dá forma (batch, 1, cnn_dim_output)
            image_features_before = image_features.unsqueeze(1)

        # Projeção p/ espaço comum
        b_i, s_i, d_i = image_features_before.shape
        image_features_before = image_features_before.view(b_i*s_i, d_i)
        projected_image_features = self.image_projector(image_features_before)
        image_features = projected_image_features.view(b_i, s_i, -1)
        # -> (seq_len_img, batch, common_dim)
        image_features = image_features.permute(1, 0, 2)

        # === [B] Extrator de Texto ===
        if self.text_model_name in ["one-hot-encoder", "tab-transformer"]:
            text_features = self.text_fc(text_metadata)  # (batch, 512)
            text_features = text_features.unsqueeze(1) # Adiciona uma dimensão às features

            # Projeção para espaço comum
            b_tt, s_tt, d_tt = text_features.shape
            before_project_text_features = text_features.view(b_tt*s_tt, d_tt)
            projected_text_features = self.text_projector(before_project_text_features)
            text_features = projected_text_features.view(b_tt, s_tt, -1)
            text_features = text_features.permute(1, 0, 2)

        elif (self.text_model_name in ["gpt2", "bert-base-uncased"]):
            # Ajustar o formato de input_ids e attention_mask
            input_ids = text_metadata['input_ids'].squeeze(1)
            attention_mask = text_metadata['attention_mask'].squeeze(1)

            # Extração de características textuais
            if "gpt2" in self.text_model_name.lower():
                text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                # A saída do GPT-2
                text_features = text_outputs.last_hidden_state[:, -1, :]  # Último token da sequência
                # Alternativamente, se você quiser usar a média de todos os tokens:
                # text_features = text_outputs.last_hidden_state.mean(dim=1)
            else:
                text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_features = text_outputs.last_hidden_state[:, 0, :]  # Usar token [CLS] para BERT

            # Movendo para o dispositivo correto
            text_features = text_features.to(self.device)
            text_features = text_features.unsqueeze(1)
            before_project_text_features = text_features.permute(1, 0, 2)
            # Projeção para espaço comum
            projected_text_features = self.text_projector(before_project_text_features)
            text_features = projected_text_features
        else:
            raise ValueError("Encoder de texto não implementado!\n")
        
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

        if self.attention_mecanism=="no-metadata":
            if self.cnn_model_name in ["google/vit-base-patch16-224","openai/clip-vit-base-patch16", "facebookresearch/dinov2"]:
                # Os modelos ViT possuem uma sequência de tokens que precisa ser processada antes de ser projetada
                projected_image_features = projected_image_features.view(b_i, s_i, -1).mean(dim=1)  # (batch, common_dim)
            combined_features = projected_image_features
            output = self.fc_fusion(combined_features)  # (batch, num_classes)

            return output
        elif self.attention_mecanism=="no-metadata-without-mlp":
            output = self.fc_no_mlp_to_visual_cls(image_features_before)
            return output
        
        elif self.attention_mecanism=="metablock":
            meta_block_features = self.meta_block(image_features_before, before_project_text_features)  # [B, num_channels, H', W']
            # Pooling global e classificação
            pooled_features = self.avg_pool(meta_block_features)  # [B, num_channels, 1, 1]
            pooled_features = pooled_features.view(pooled_features.size(0), -1)  # [B, num_channels]
            return self.fc_no_mlp_to_visual_cls(pooled_features)
        
        elif self.attention_mecanism == "weighted":
            # # === [F] Gating: quanto usar de 'peso' para cada modal
            if self.cnn_model_name in ["google/vit-base-patch16-224","openai/clip-vit-base-patch16", "facebookresearch/dinov2"]:
                # Os modelos ViT possuem uma sequência de tokens que precisa ser processada antes de ser projetada
                projected_image_features = projected_image_features.view(b_i, s_i, -1).mean(dim=1)  # (batch, common_dim)
                projected_text_features = projected_text_features.view(b_tt, s_tt, -1).mean(dim=1)  # (batch, common_dim)

            projected_image_features = projected_image_features.squeeze(0)
            projected_text_features = projected_text_features.squeeze(0)
            alpha_img = torch.sigmoid(self.img_gate(projected_image_features))  # (batch, common_dim)
            alpha_txt = torch.sigmoid(self.txt_gate(projected_text_features))   # (batch, common_dim)

            # Multiplicamos as features pela máscara gerada
            image_pooled_gated = alpha_img * projected_image_features
            text_pooled_gated = alpha_txt * projected_text_features
            combined_features = torch.cat([image_pooled_gated, text_pooled_gated], dim=1)
        elif self.attention_mecanism == "concatenation":
            if self.cnn_model_name in ["google/vit-base-patch16-224","openai/clip-vit-base-patch16", "facebookresearch/dinov2"]:
                # Os modelos ViT possuem uma sequência de tokens que precisa ser processada antes de ser projetada
                projected_image_features = projected_image_features.view(b_i, s_i, -1).mean(dim=1)  # (batch, common_dim)
                projected_text_features = projected_text_features.view(b_tt, s_tt, -1).mean(dim=1)  # (batch, common_dim)
            # # Apenas concatena as features projetadas
            combined_features = torch.cat((projected_image_features.squeeze(0), projected_text_features.squeeze(0)), dim=-1)
        elif self.attention_mecanism == "weighted-after-crossattention":
            # # === [F] Gating: quanto usar de cada modal?
            #  Após o uso de cross-attention, as features são multiplicadas por cada fator individual de cada modalidade
            alpha_img = torch.sigmoid(self.img_gate(image_pooled))  # (batch, common_dim)
            alpha_txt = torch.sigmoid(self.txt_gate(text_pooled))   # (batch, common_dim)
            # Multiplicamos as features pela máscara gerada
            image_pooled_gated = alpha_img * image_pooled
            text_pooled_gated = alpha_txt * text_pooled
            # === [G] Fusão e classificação
            combined_features = torch.cat([image_pooled_gated, text_pooled_gated], dim=1)
        elif self.attention_mecanism == "crossattention":
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)
        elif self.attention_mecanism == "cross-weights-after-crossattention":
            #  Após o uso de cross-attention, as features são multiplicadas por cada fator individual de cada modalidade
            alpha_img = torch.sigmoid(self.img_gate(image_pooled))  # (batch, common_dim)
            alpha_txt = torch.sigmoid(self.txt_gate(text_pooled))   # (batch, common_dim)
            # Multiplicamos as features pela máscara gerada
            image_pooled_gated = alpha_txt * image_pooled
            text_pooled_gated = alpha_img * text_pooled
            # === [G] Fusão e classificação
            combined_features = torch.cat([image_pooled_gated, text_pooled_gated], dim=1)

        # Com blocos residuais e gated
        elif self.attention_mecanism=="att-intramodal+residual":
            # === Self-Attention Intra-Modality ===
            image_features_att, _ = self.image_self_attention(
                image_features, image_features, image_features
            )

            text_features_att, _ = self.text_self_attention(
                text_features, text_features, text_features
            )
            # Bloco residual
            image_features_residual_before_cross_attention = self.image_residual(image_features, image_features_att, image_features_att)
            text_features_residual_before_cross_attention = self.text_residual(text_features, text_features_att, text_features_att)
            
            # === Pooling das features finais 
            image_features_residual_before_cross_attention = image_features_residual_before_cross_attention.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
            text_features_residual_before_cross_attention = text_features_residual_before_cross_attention.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

            image_pooled = image_features_residual_before_cross_attention.mean(dim=1)  # (batch, common_dim)
            text_pooled = text_features_residual_before_cross_attention.mean(dim=1)    # (batch, common_dim)
            # === Fusão das features
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)
        elif self.attention_mecanism=="att-intramodal+residual+cross-attention-metadados":
            # === Self-Attention Intra-Modality ===
            image_features_att, _ = self.image_self_attention(
                image_features, image_features, image_features
            )

            text_features_att, _ = self.text_self_attention(
                text_features, text_features, text_features
            )
            # Bloco residual
            image_features_residual_before_cross_attention = self.image_residual(image_features, image_features_att, image_features_att)
            text_features_residual_before_cross_attention = self.text_residual(text_features, text_features_att, text_features_att)

            # === Cross-Attention Inter-Modality ===
            image_cross_att, _ = self.image_cross_attention(
                query=image_features_residual_before_cross_attention,
                key=text_features_residual_before_cross_attention,
                value=text_features_residual_before_cross_attention
            )
            # "Texto assiste à imagem"
            text_cross_att, _ = self.text_cross_attention(
                query=text_features_residual_before_cross_attention,
                key=image_features_residual_before_cross_attention,
                value=image_features_residual_before_cross_attention
            )
            # === Pooling das features finais 
            image_cross_att = image_cross_att.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
            text_cross_att = text_cross_att.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

            image_pooled = image_cross_att.mean(dim=1)  # (batch, common_dim)
            text_pooled = text_cross_att.mean(dim=1)    # (batch, common_dim)
            # === Fusão das features
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)

        elif self.attention_mecanism=="att-intramodal+residual+cross-attention-metadados+att-intramodal+residual":
            # === Self-Attention Intra-Modality ===
            image_features_att, _ = self.image_self_attention(
                image_features, image_features, image_features
            )

            text_features_att, _ = self.text_self_attention(
                text_features, text_features, text_features
            )
            # Bloco residual
            image_features_residual_before_cross_attention = self.image_residual(image_features, image_features_att, image_features_att)
            text_features_residual_before_cross_attention = self.text_residual(text_features, text_features_att, text_features_att)

            # === Cross-Attention Inter-Modality ===
            image_cross_att, _ = self.image_cross_attention(
                query=image_features_residual_before_cross_attention,
                key=text_features_residual_before_cross_attention,
                value=text_features_residual_before_cross_attention
            )
            # "Texto assiste à imagem"
            text_cross_att, _ = self.text_cross_attention(
                query=image_features_residual_before_cross_attention,
                key=image_features_residual_before_cross_attention,
                value=text_features_residual_before_cross_attention
            )

            # === Self-Attention Intra-Modality after cross-attention ===
            image_features_att_after_cross_att, _ = self.image_self_attention(
                image_cross_att, image_cross_att, image_cross_att
            )

            text_features_att_after_cross_att, _ = self.text_self_attention(
                text_cross_att, text_cross_att, text_cross_att
            )
            # Bloco residual
            image_features_residual_after_cross_attention = self.image_residual(image_features, image_features_att_after_cross_att, image_features_att_after_cross_att)
            text_features_residual_after_cross_attention = self.text_residual(text_features, text_features_att_after_cross_att, text_features_att_after_cross_att)
            
            # === Pooling das features finais 
            image_features_residual_after_cross_attention = image_features_residual_after_cross_attention.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
            text_features_residual_after_cross_attention = text_features_residual_after_cross_attention.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

            image_pooled = image_features_residual_after_cross_attention.mean(dim=1)  # (batch, common_dim)
            text_pooled = text_features_residual_after_cross_attention.mean(dim=1)    # (batch, common_dim)
            # === Fusão das features
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)
        output = self.fc_fusion(combined_features)  # (batch, num_classes)
        return output