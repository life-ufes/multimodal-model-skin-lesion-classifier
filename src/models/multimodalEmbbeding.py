import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalModel, self).__init__()
        self.cnn_dim_output = 2048
        self.text_encoder_dim_output = 1024
        self.attention_heads = 8
        
        # **CNN para Imagens**
        self.cnn = models.resnet50(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False  # Congelar pesos da ResNet50
        self.cnn.fc = nn.Identity()  # Remover camada final (2048 saídas)

        # **BERT para Metadados Textuais**
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False  # Congelar pesos do BERT

        # Reduzir dimensionalidade dos embeddings do BERT
        self.bert_fc = nn.Sequential(
            nn.Linear(768, self.text_encoder_dim_output),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # **Camada Final Combinada**
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_dim_output + self.text_encoder_dim_output, 2048),
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

        # Camada de Atenção
        self.attention = nn.MultiheadAttention(embed_dim=int(self.cnn_dim_output + self.text_encoder_dim_output), num_heads=self.attention_heads, dropout=0.3)


    def forward(self, image, tokenized_text):
        # Extrair recursos da imagem
        image_features = self.cnn(image)

        # Processar texto com BERT
        bert_output = self.bert(
            input_ids=tokenized_text['input_ids'].squeeze(1),  # Remove batch dimension extra
            attention_mask=tokenized_text['attention_mask'].squeeze(1)
        )
        text_features = bert_output.pooler_output  # Vetor de saída (768 dimensões)
        text_features = self.bert_fc(text_features)

        # Combinar tudo
        combined_features = torch.cat((image_features, text_features), dim=1)
         # Ajuste de dimensões para a camada de atenção
        combined_features = combined_features.unsqueeze(0)  # (1, batch_size, 2112)
        
        # Aplicar atenção (Nota: para a MultiheadAttention, a entrada precisa ser (seq_len, batch_size, embed_dim))
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)  # (1, batch_size, 2112)
        
        # Remover a dimensão adicional
        attn_output = attn_output.squeeze(0)  # (batch_size, 2112)

        # Passar pelas camadas finais
        return self.fc(attn_output)