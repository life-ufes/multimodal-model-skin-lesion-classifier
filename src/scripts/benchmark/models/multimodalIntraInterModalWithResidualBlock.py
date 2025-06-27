import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformers import ViTFeatureExtractor, CLIPProcessor, CLIPModel

from loadImageModelClassifier import loadModels

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=False),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        residual = x
        out, _ = self.layer(x, x, x)
        out = out + residual
        return out

class BilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(BilinearPooling, self).__init__()
        self.fc1 = nn.Linear(input_dim1, output_dim)
        self.fc2 = nn.Linear(input_dim2, output_dim)
    
    def forward(self, x1, x2):
        out1 = self.fc1(x1)
        out2 = self.fc2(x2)
        out = out1 * out2  # Multiplicação elemento a elemento
        return out

class MetaBlock(nn.Module):
    def __init__(self, metadata_dim, image_feature_dim):
        """
        Args:
            metadata_dim (int): Dimensão dos metadados de entrada.
            image_feature_dim (int): Dimensão das features de imagem.
        """
        super(MetaBlock, self).__init__()
        self.f_b = nn.Linear(metadata_dim, image_feature_dim)
        self.g_b = nn.Linear(metadata_dim, image_feature_dim)
    
    def forward(self, image_features, metadata):
        """
        Args:
            image_features (Tensor): Features de imagem, shape (batch_size, image_feature_dim)
            metadata (Tensor): Metadados, shape (batch_size, metadata_dim)
        
        Returns:
            Tensor: Features de imagem modificadas, shape (batch_size, image_feature_dim)
        """
        # Calcula f_b(X_meta) e g_b(X_meta)
        f_b = self.f_b(metadata)  # (batch_size, image_feature_dim)
        g_b = self.g_b(metadata)  # (batch_size, image_feature_dim)
        
        # Aplica tanh e sigmoid
        T_gate = torch.tanh(f_b) * image_features  # (batch_size, image_feature_dim)
        S_gate = torch.sigmoid(T_gate) + g_b      # (batch_size, image_feature_dim)
        
        # Modifica as features de imagem
        modified_image_features = image_features * S_gate  # (batch_size, image_feature_dim)
        
        return modified_image_features

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
        # 3) MetaBlock
        # -------------------------
        metadata_dim = vocab_size if self.text_model_name == "one-hot-encoder" else self.text_encoder_dim_output
        self.metablock = MetaBlock(metadata_dim, self.common_dim)

        # -------------------------
        # 4) Atenções Intra e Inter
        # -------------------------
        self.image_self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=False  # Padrão: [seq_len, batch, embed_dim]
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
        # 5) Gating Mechanisms
        # -------------------------
        self.img_gate = nn.Linear(self.common_dim, self.common_dim)
        self.txt_gate = nn.Linear(self.common_dim, self.common_dim)

        # -------------------------
        # 6) Residual Blocks
        # -------------------------
        self.image_residual = ResidualBlock(self.common_dim)
        self.text_residual = ResidualBlock(self.common_dim)

        # -------------------------
        # 7) Fusões de Features Avançadas
        # -------------------------
        self.bilinear_pool = BilinearPooling(self.common_dim, self.common_dim, self.common_dim)

        # -------------------------
        # 8) Camada de Fusão Final
        # -------------------------
        # Ajustar a dimensão de entrada da primeira camada Linear com base no mecanismo de atenção
        if self.attention_mecanism == "concatenation":
            fusion_input_dim = self.common_dim * 2  # Concatenando imagem e texto
        else:
            fusion_input_dim = self.common_dim  # Outros mecanismos, como weighted

        self.fc_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.common_dim),
            nn.BatchNorm1d(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.BatchNorm1d(self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim // 2, num_classes),
            nn.Softmax(dim=1)  # Remover se usar CrossEntropyLoss ou Focal Loss
        )
    
    def forward(self, image, text_metadata):
        """
        image: tensor de imagens (batch, C, H, W) se CNN 
               ou lista PIL se ViTFeatureExtractor
        text_metadata: dicionário c/ 'input_ids', 'attention_mask' (BERT/Bart)
                       ou tensor se "one-hot-encoder".
        """

        # === [A] Extrator de Imagem ===
        if self.cnn_model_name in ["vit-base-patch16-224", "openai/clip-vit-base-patch16"]:
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            outputs = self.image_encoder(**inputs)
            # outputs.last_hidden_state => (batch, seq_len_img, hidden_dim)
            image_features = outputs.last_hidden_state
        else:
            # CNN -> (batch, cnn_dim_output)
            image_features = self.image_encoder(image)
            # Dá forma (batch, 1, cnn_dim_output)
            image_features = image_features.unsqueeze(1)

        # Projeção para espaço comum
        b_i, s_i, d_i = image_features.shape  # (batch, seq_len_img, cnn_dim_output)
        image_features = image_features.view(b_i * s_i, d_i)  # (batch * seq_len_img, cnn_dim_output)
        projected_image_features = self.image_projector(image_features)  # (batch * seq_len_img, common_dim)
        image_features = projected_image_features.view(b_i, s_i, -1)  # (batch, seq_len_img, common_dim)
        # -> (seq_len_img, batch, common_dim)
        image_features = image_features.permute(1, 0, 2)  # (seq_len_img, batch, common_dim)

        # === [B] Extrator de Texto ===
        if self.text_model_name == "one-hot-encoder":
            text_features = self.text_fc(text_metadata)  # (batch, 512)
            text_features = text_features.unsqueeze(1)  # (batch, 1, 512)
        else:
            # Ajustar input_ids e attention_mask para shape [batch, seq_len]
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
            text_features = text_features.view(b_t * s_t, d_t)  # (batch * seq_len_text, 768)
            text_features = self.text_fc(text_features)  # (batch * seq_len_text, 512)
            text_features = text_features.view(b_t, s_t, -1)  # (batch, seq_len_text, 512)

        # Projeção para espaço comum
        b_tt, s_tt, d_tt = text_features.shape  # (batch, seq_len_text, 512)
        text_features = text_features.view(b_tt * s_tt, d_tt)  # (batch * seq_len_text, 512)
        projected_text_features = self.text_projector(text_features)  # (batch * seq_len_text, 512)
        text_features = projected_text_features.view(b_tt, s_tt, -1)  # (batch, seq_len_text, 512)
        text_features = text_features.permute(1, 0, 2)  # (seq_len_text, batch, 512)

        # === [C] MetaBlock ===
        # Pooling das features de texto para gerar um vetor representativo para o MetaBlock
        text_pooled = text_features.mean(dim=0)  # (batch, common_dim)

        # Ajustar as features de imagem para [batch, common_dim]
        image_pooled = image_features.mean(dim=0)  # (batch, common_dim)

        # Aplicar MetaBlock nas features de imagem usando os metadados (text_pooled)
        modified_image_features = self.metablock(image_pooled, projected_text_features)  # (batch, common_dim)

        # Replicar para seq_len_img
        modified_image_features = modified_image_features.unsqueeze(1).repeat(1, s_i, 1)  # (batch, seq_len_img, common_dim)

        # Permutar para [seq_len_img, batch, common_dim]
        modified_image_features = modified_image_features.permute(1, 0, 2)  # (seq_len_img, batch, common_dim)

        # === [D] Self-Attention Intra-Modality ===
        image_features_att, _ = self.image_self_attention(
            modified_image_features, modified_image_features, modified_image_features
        )
        image_features_att = self.image_residual(image_features_att)  # Residual Connection

        text_features_att, _ = self.text_self_attention(
            text_features, text_features, text_features
        )
        text_features_att = self.text_residual(text_features_att)  # Residual Connection

        # === [E] Cross-Attention Inter-Modality ===
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

        # === [F] Pooling das atenções finais 
        image_cross_att = image_cross_att.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
        text_cross_att = text_cross_att.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

        image_pooled = image_cross_att.mean(dim=1)  # (batch, common_dim)
        text_pooled = text_cross_att.mean(dim=1)    # (batch, common_dim)

        if self.attention_mecanism == "weighted":
            # Verificar se usa VIT
            if self.cnn_model_name == "vit-base-patch16-224":
                # As features de imagem já foram processadas
                pass  # Nenhuma ação adicional necessária
            # Gating: quanto usar de 'peso' para cada modal
            alpha_img = torch.sigmoid(self.img_gate(image_pooled))  # (batch, common_dim)
            alpha_txt = torch.sigmoid(self.txt_gate(text_pooled))   # (batch, common_dim)
            
            # Multiplicamos as features pela máscara gerada
            image_pooled_gated = alpha_img * image_pooled
            text_pooled_gated = alpha_txt * text_pooled
            combined_features = torch.cat([image_pooled_gated, text_pooled_gated], dim=1)  # (batch, 2 * common_dim)

        elif self.attention_mecanism == "concatenation":
            # Concatenar as features de imagem e texto
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)  # (batch, 2 * common_dim)

        elif self.attention_mecanism == "gfcam":
            # Gating após cross-attention
            alpha_img = torch.sigmoid(self.img_gate(image_pooled))  # (batch, common_dim)
            alpha_txt = torch.sigmoid(self.txt_gate(text_pooled))   # (batch, common_dim)

            # Multiplicamos as features pela máscara gerada
            image_pooled_gated = alpha_img * image_pooled
            text_pooled_gated = alpha_txt * text_pooled

            # Fusão
            combined_features = torch.cat([image_pooled_gated, text_pooled_gated], dim=1)  # (batch, 2 * common_dim)

        elif self.attention_mecanism == "crossattention":
            # Apenas concatenar as features de cross-attention
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)  # (batch, 2 * common_dim)

        else:
            raise ValueError(f"Attention mechanism '{self.attention_mecanism}' not supported.")

        # === [G] Fusão e Classificação ===
        output = self.fc_fusion(combined_features)  # (batch, num_classes)
        return output
