import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loadImageModelClassifier import loadModels
from transformers import ViTFeatureExtractor

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device, cnn_model_name, text_model_name, vocab_size=76):
        super(MultimodalModel, self).__init__()
        
        # Dimensões do modelo
        self.common_dim = 512
        self.text_encoder_dim_output = 512
        self.cnn_dim_output = 512
        self.device = device
        self.feature_extractor = None
        self.cnn_model_name = cnn_model_name
        self.text_model_name = text_model_name

        # Carregar o modelo extrator de features das imagens
        self.image_encoder, self.cnn_dim_output = loadModels.loadModelImageEncoder(self.cnn_model_name, self.common_dim)

        # Carregar o extrator de características e o modelo ViT
        if self.cnn_model_name=="vit-base-patch16-224":
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(f"google/{self.cnn_model_name}")        
        
        ## Carregar o modelo extrator de features dos textos
        if self.text_model_name=="one-hot-encoder":
            # Projeção para os metadados (pré-processados)
            self.text_fc = nn.Sequential(
                nn.Linear(vocab_size, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, self.text_encoder_dim_output)
            )
        else: 
            self.text_encoder, self.text_encoder_dim_output = loadModels.loadTextModelEncoder(text_model_name)
            # Reduzir dimensionalidade dos embeddings do tipo BERT
            self.text_fc = nn.Sequential(
                nn.Linear(768, self.text_encoder_dim_output),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        # Projeções para o espaço comum
        self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)
        # Projeção para os metadados
        self.text_projector = nn.Linear(self.text_encoder_dim_output, self.common_dim)
        
        # Self-Attention Intra-Modality
        self.image_self_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=8)
        self.text_self_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=8)
        
        # Cross-Attention Inter-Modality
        self.image_cross_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        self.text_cross_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        
        # Camada de fusão
        self.fc_fusion = nn.Sequential(
            nn.Linear(self.common_dim * 2, 1024),
            nn.BatchNorm1d(1024), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, image, text_metadata):
        if self.cnn_model_name=="vit-base-patch16-224":
            # Pré-processando as imagens
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            
            # Passando as imagens pelo modelo ViT
            outputs = self.image_encoder(**inputs)
            
            # Pegando a saída da última camada (features do token [CLS])
            image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        else:
            image_features = self.image_encoder(image)
        # Projeção para o espaço comum
        image_features = self.image_projector(image_features)
        image_features = image_features.unsqueeze(1)  # Shape: (batch_size, seq_length=1, common_dim)
        image_features = image_features.permute(1, 0, 2)  # Shape: (seq_length=1, batch_size, common_dim)
        
        if self.text_model_name =="one-hot-encoder":
            # Projeção de metadados já pré-processados
            text_features = self.text_fc(text_metadata)  # Saída: (batch_size, common_dim)
        else:
            # Processar texto com BERT
            text_encoder_output = self.text_encoder(
                input_ids=text_metadata['input_ids'].squeeze(1),
                attention_mask=text_metadata['attention_mask'].squeeze(1)
            )

            text_features = text_encoder_output.last_hidden_state[:, 0, :]  # Vetor [CLS] (primeiro token) para classificação
            text_features = self.text_fc(text_features)
        
        # Obter as features com as mesmas dimensões
        text_features = self.text_projector(text_features)
        text_features = text_features.unsqueeze(1)  # Shape: (batch_size, seq_length=1, common_dim)
        text_features = text_features.permute(1, 0, 2)  # Shape: (seq_length=1, batch_size, common_dim)
        
        # Self-Attention intra-modalidade
        image_features_att, _ = self.image_self_attention(image_features, image_features, image_features)
        text_features_att, _ = self.text_self_attention(text_features, text_features, text_features)
        
        # Cross-Attention inter-modalidade
        image_cross_att, _ = self.image_cross_attention(image_features_att, text_features_att, text_features_att)
        text_cross_att, _ = self.text_cross_attention(text_features_att, image_features_att, image_features_att)
        
        # Combinação das características
        combined_features = torch.cat([image_cross_att.squeeze(0), text_cross_att.squeeze(0)], dim=1)
        
        # Classificação final
        output = self.fc_fusion(combined_features)
        return output
