import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Classe para Embedding de Texto utilizando BERT
class TextEmbedding(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(TextEmbedding, self).__init__()
        # Inicialize o tokenizer e o modelo BERT pré-treinado
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name)

    def forward(self, text):
        # Tokenizar a entrada de texto e retornar tensores para o modelo BERT
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Passar pelos embeddings do BERT
        outputs = self.bert(**inputs)

        # Pegue a média das representações das palavras da sequência
        return outputs.last_hidden_state.mean(dim=1)  # Média das representações de todas as palavras
