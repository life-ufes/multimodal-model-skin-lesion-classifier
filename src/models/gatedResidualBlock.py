import torch
import torch.nn as nn

class GatedAlteredResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(GatedAlteredResidualBlock, self).__init__()
        # Pré-normalização da query
        self.norm = nn.LayerNorm(dim)
        # Atenção multi-cabeça com número reduzido de cabeças (ex.: 8)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=False)
        self.dropout1 = nn.Dropout(dropout)
        # Mecanismo de gating para a saída da atenção
        self.gate_linear = nn.Linear(dim, dim)
        
     
    
    def forward(self, q, k, v):
        # Pré-normalização da query
        # q_norm = self.norm1(q)
        # Cálculo da atenção
        attn_output, _ = self.attn(q, k, v)
        attn_output = self.dropout1(attn_output)
        # Geração do gating a partir da query original
        gate = torch.sigmoid(self.gate_linear(q))
        # Conexão residual com gating: a porta decide a contribuição da saída da atenção
        out = q + gate * attn_output
        
       
        # Conexão residual com normalização final
        out2 = self.norm(out)
        return out2