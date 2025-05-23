import torch
import torch.nn as nn

class SimpleAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(SimpleAttentionBlock, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        # Camada de Atenção Multi-Cabeças
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)

        # Dropout para regulação
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor de entrada com forma [seq_len, batch_size, input_dim]
        """
        # Atenção (aqui usamos a entrada como chave, valor e consulta)
        attn_output, _ = self.attention(x, x, x)

        # Aplicando dropout para regulação
        output = self.dropout(attn_output)

        return output
