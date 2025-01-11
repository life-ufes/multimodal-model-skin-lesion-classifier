import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel


class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device, cnn_model_name, text_model_name):
        super(MultimodalModel, self).__init__()
        self.device = device
        
        # Definir encoder de imagem
        if cnn_model_name == "resnet-50":
            self.cnn = models.resnet50(pretrained=True)
            cnn_output_dim = 2048
        elif cnn_model_name == "resnet-18":
            self.cnn = models.resnet18(pretrained=True)
            cnn_output_dim = 512
        else:
            raise ValueError("Modelo de CNN não suportado.")
        
        # Congelar parâmetros da CNN
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.fc = nn.Identity()  # Remover camada de classificação
        
        # Definir encoder de texto (BERT)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        bert_output_dim = self.text_encoder.config.hidden_size
        
        # Camadas de atenção intra-modular
        self.visual_attention = nn.MultiheadAttention(embed_dim=cnn_output_dim, num_heads=4)
        self.text_attention = nn.MultiheadAttention(embed_dim=bert_output_dim, num_heads=4)
        
        # Camada de fusão e classificação
        self.fc = nn.Linear(cnn_output_dim + bert_output_dim, num_classes)
    
    def forward(self, image, metadata):
        # Extração de características visuais
        image_features = self.cnn(image)  # Saída da CNN

        # Atenção intra-modular para características visuais
        image_features = image_features.unsqueeze(0)  # Adicionar dimensão de sequência
        image_attention_output, _ = self.visual_attention(image_features, image_features, image_features)
        image_attention_output = image_attention_output.squeeze(0)  # Remover dimensão de sequência

        # Ajustar o formato de input_ids e attention_mask
        input_ids = metadata['input_ids'].squeeze(1)
        attention_mask = metadata['attention_mask'].squeeze(1)

        # Extração de características textuais
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # Usar token [CLS]

        # Atenção intra-modular para características textuais
        text_features = text_features.unsqueeze(0)  # Adicionar dimensão de sequência
        text_attention_output, _ = self.text_attention(text_features, text_features, text_features)
        text_attention_output = text_attention_output.squeeze(0)  # Remover dimensão de sequência

        # Combinação das saídas
        combined_features = torch.cat((image_attention_output, text_attention_output), dim=-1)

        # Classificação final
        outputs = self.fc(combined_features)

        return outputs
