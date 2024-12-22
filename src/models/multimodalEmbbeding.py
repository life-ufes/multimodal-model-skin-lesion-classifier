import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer

# Classe para Embedding de Texto utilizando BERT
class TextEmbedding(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(TextEmbedding, self).__init__()
        # Inicialize o tokenizer e o modelo BERT pré-treinado
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name)

    def forward(self, text):
        # Certifique-se de que text seja uma lista de strings
        if isinstance(text, str):
            text = [text]  # Se for um único texto, coloque-o em uma lista
        # Tokenizar a entrada de texto e retornar tensores para o modelo BERT
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Passar pelos embeddings do BERT
        outputs = self.bert(**inputs)

        # Pegue a média das representações das palavras da sequência
        return outputs.last_hidden_state.mean(dim=1)  # Média das representações de todas as palavras

# Modelo Multimodal com BERT para codificar os dados textuais
class MultimodalModelWithEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim=64, pretrained_model_name='bert-base-uncased'):
        super(MultimodalModelWithEmbedding, self).__init__()

        # CNN para imagens (ResNet50 exemplo)
        self.cnn = models.resnet50(pretrained=True)

        # Congelar os pesos da ResNet50
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Substituir a camada final por uma identidade
        self.cnn.fc = nn.Identity()

        # Embedding de texto utilizando BERT
        self.text_embedding = TextEmbedding(pretrained_model_name)

        # Camada totalmente conectada para a combinação de features
        self.fc = nn.Sequential(
            nn.Linear(2048 + embedding_dim, 1024),  # Features da imagem + Embedding de texto
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)  # Softmax na saída para classificação
        )

    def forward(self, image_features, text_data):
        # Passar os dados textuais para obter os embeddings
        text_embeddings = self.text_embedding(text_data)

        # Concatenar as features das imagens e os embeddings do texto
        combined_features = torch.cat((image_features, text_embeddings), dim=1)

        # Passar pela camada totalmente conectada para a saída
        output = self.fc(combined_features)

        return output
