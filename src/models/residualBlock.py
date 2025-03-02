import torch 
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=512, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.tanh = nn.Tanh()
    
    def forward(self, q, k, v):
        residual = q
        attn_output, _ = self.attn(q, k, v)
        attn_output = self.dropout(attn_output)
        attn_output = self.tanh(attn_output)
        out = self.norm(attn_output + residual)
        return out
