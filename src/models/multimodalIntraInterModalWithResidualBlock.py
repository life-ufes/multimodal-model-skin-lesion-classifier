import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, CLIPProcessor, CLIPModel
from loadImageModelClassifier import loadModels
from models.focalLoss import FocalLoss  # Assegure-se de que a FocalLoss está corretamente importada

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True),
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
        out = out1 * out2  # Element-wise multiplication
        return out

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
        self.image_projector_dropout = nn.Dropout(0.3)
        
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
        self.text_projector_dropout = nn.Dropout(0.3)
        
        # -------------------------
        # 3) Atenções Intra e Inter
        # -------------------------
        self.image_self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=True  # Alterado para True para compatibilidade com ResidualBlock
        )
        self.text_self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=True
        )
        
        self.image_cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=True
        )
        self.text_cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Adicionar blocos residuais
        self.image_residual = ResidualBlock(self.common_dim)
        self.text_residual = ResidualBlock(self.common_dim)
        
        # -------------------------
        # 4) Fusões de Features Avançadas
        # -------------------------
        self.bilinear_pool = BilinearPooling(self.common_dim, self.common_dim, self.common_dim)
        
        # -------------------------
        # 5) Camada de Fusão Final
        # -------------------------
        self.fc_fusion = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.BatchNorm1d(self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim // 2, num_classes),
            # Remover Softmax para uso com CrossEntropyLoss/Focal Loss
        )
    
    def forward(self, image, text_metadata):
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
        
        # Projeção p/ espaço comum
        image_features = self.image_projector(image_features)  # (batch, seq_len_img, common_dim)
        image_features = self.image_projector_dropout(image_features)
        
        # === [B] Extrator de Texto ===
        if self.text_model_name == "one-hot-encoder":
            text_features = self.text_fc(text_metadata)  # (batch, 512)
            text_features = text_features.unsqueeze(1) # (batch, 1, 512)
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

            text_features = self.text_fc(text_features)  # (batch, seq_len_text, 512)
        
        # Projeção para espaço comum
        text_features = self.text_projector(text_features)  # (batch, seq_len_text, common_dim)
        text_features = self.text_projector_dropout(text_features)
        
        # === [C] Self-Attention Intra-Modality ===
        image_features_att, _ = self.image_self_attention(
            image_features, image_features, image_features
        )
        image_features_att = self.image_residual(image_features_att)  # Residual Connection
        
        text_features_att, _ = self.text_self_attention(
            text_features, text_features, text_features
        )
        text_features_att = self.text_residual(text_features_att)  # Residual Connection
        
        # === [D] Cross-Attention Inter-Modality ===
        image_cross_att, _ = self.image_cross_attention(
            query=image_features_att,
            key=text_features_att,
            value=text_features_att
        )
        text_cross_att, _ = self.text_cross_attention(
            query=text_features_att,
            key=image_features_att,
            value=image_features_att
        )
        
        # === [E] Pooling das atenções finais 
        image_pooled = image_cross_att.mean(dim=1)  # (batch, common_dim)
        text_pooled = text_cross_att.mean(dim=1)    # (batch, common_dim)
        
        # === [F] Fusões de Features Avançadas ===
        combined_features = self.bilinear_pool(image_pooled, text_pooled)  # (batch, common_dim)
        
        # === [G] Fusão e Classificação ===
        output = self.fc_fusion(combined_features)  # (batch, num_classes)
        return output
