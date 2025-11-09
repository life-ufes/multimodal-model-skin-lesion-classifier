import torch.nn as nn
import torch

class MetaBlock(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U):
        super(MetaBlock, self).__init__()
        self.fb = nn.Sequential(
            nn.Linear(U, V),
            nn.LayerNorm(V)    # ✅ Corrigido
        )
        self.gb = nn.Sequential(
            nn.Linear(U, V),
            nn.LayerNorm(V)    # ✅ Corrigido
        )

    def forward(self, V, U):
        t1 = self.fb(U)        # [B, V]
        t2 = self.gb(U)        # [B, V]
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V
