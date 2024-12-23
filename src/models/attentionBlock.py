import torch
import torch.nn as nn
class TransformerAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(TransformerAttentionBlock, self).__init__()
        assert input_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Atenção
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(feedforward_output))
        
        return x
