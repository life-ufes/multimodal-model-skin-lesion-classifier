import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, cnn_model_name):
        super(MultimodalModel, self).__init__()
        self.cnn_model_name = cnn_model_name
        self.cnn_dim_output = 512  # ResNet18 
        self.text_encoder_dim_output = 1024
        self.common_dim = 1024  # Dimensão comum para combinação
        self.attention_heads = 8
        
        # **CNN para Imagens**
        if self.cnn_model_name=="resnet-50":        
            self.cnn = models.resnet50(pretrained=True)
            self.cnn_dim_output = 2048

        if self.cnn_model_name=="resnet-18":
            self.cnn = models.resnet18(pretrained=True)
            self.cnn_dim_output = 512

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.fc = nn.Identity()  # Remover camada final (2048 saídas)

        # **BERT para Metadados Textuais**
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        # Reduzir dimensionalidade dos embeddings do BERT
        self.bert_fc = nn.Sequential(
            nn.Linear(768, self.text_encoder_dim_output),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Projeções para espaço comum
        self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)
        self.text_projector = nn.Linear(self.text_encoder_dim_output, self.common_dim)

        # Mecanismo de Atenção Cruzada
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=self.attention_heads, dropout=0.3)

        # **Camada Final Combinada**
        self.fc = nn.Sequential(
            nn.Linear(self.common_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
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

    def forward(self, image, tokenized_text):
        # Extrair recursos da imagem
        image_features = self.cnn(image)
        image_features = self.image_projector(image_features)  # Projeção para dimensão comum

        # Processar texto com BERT
        bert_output = self.bert(
            input_ids=tokenized_text['input_ids'].squeeze(1),
            attention_mask=tokenized_text['attention_mask'].squeeze(1)
        )
        text_features = bert_output.pooler_output  # Vetor de saída do BERT (768 dimensões)
        text_features = self.bert_fc(text_features)
        text_features = self.text_projector(text_features)  # Projeção para dimensão comum

        # Atenção cruzada
        # Ajustar dimensões: MultiheadAttention espera (seq_len, batch_size, embed_dim)
        image_features = image_features.unsqueeze(0)  # (1, batch_size, embed_dim)
        text_features = text_features.unsqueeze(0)  # (1, batch_size, embed_dim)

        attn_output, _ = self.cross_attention(text_features, text_features, image_features)

        # Remover dimensão de sequência
        combined_features = attn_output.squeeze(0)  # (batch_size, embed_dim)

        # Passar pela cabeça final
        output = self.fc(combined_features)
        return output