import torch 
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, q, k, v):
        q_norm = self.norm(q)
        k_norm = self.norm(k)
        v_norm = self.norm(v)

        residual = q_norm
        attn_output, _ = self.attn(q_norm, k_norm, v_norm)
        attn_output = self.dropout(attn_output)
        out = attn_output+residual # self.norm(attn_output + residual)
        return out
