import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, cnn_model_name, text_model_encoder):
        super(MultimodalModel, self).__init__()
        self.cnn_model_name = cnn_model_name
        self.cnn_dim_output = 512  # ResNet18 
        self.text_encoder_dim_output = 1024
        self.common_dim = 512  # Dimensão comum para combinação
        self.attention_heads = 8
        self.text_model_encoder = text_model_encoder
        
        # **CNN para Imagens**
        if self.cnn_model_name == "resnet-50":        
            self.cnn = models.resnet50(pretrained=True)
            self.cnn_dim_output = 2048

        elif self.cnn_model_name == "resnet-18":
            self.cnn = models.resnet18(pretrained=True)
            self.cnn_dim_output = 512

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.fc = nn.Identity()  # Remover camada final (2048 saídas)

        # **Modelo BERT para Metadados Textuais**
        self.bert_model = AutoModel.from_pretrained(self.text_model_encoder)
        for param in self.bert_model.parameters():
            param.requires_grad = False

        # Reduzir dimensionalidade dos embeddings do BERT
        self.bert_fc = nn.Sequential(
            nn.Linear(768, self.text_encoder_dim_output),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Projeções para o espaço comum
        self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)
        self.text_projector = nn.Linear(self.text_encoder_dim_output, self.common_dim)


        # **Camada Final Combinada**
        self.fc = nn.Sequential(
            nn.Linear(self.common_dim+self.common_dim, 2048),
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
        bert_output = self.bert_model(
            input_ids=tokenized_text['input_ids'].squeeze(1),
            attention_mask=tokenized_text['attention_mask'].squeeze(1)
        )

        text_features = bert_output.last_hidden_state[:, 0, :]  # Vetor [CLS] (primeiro token) para classificação
        text_features = self.bert_fc(text_features)
        text_features = self.text_projector(text_features)  # Projeção para dimensão comum

        # Atenção cruzada
        image_features = image_features.squeeze(0)  
        text_features = text_features.squeeze(0) 

        combined_features = torch.cat((image_features, text_features), dim=1)

        # Passar pela cabeça final
        output = self.fc(combined_features)
        return output
