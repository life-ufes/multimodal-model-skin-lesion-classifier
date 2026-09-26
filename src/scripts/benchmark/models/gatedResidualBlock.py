import torch
import torch.nn as nn


class GatedAlteredResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(GatedAlteredResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.gate_linear = nn.Linear(dim, dim)

    def forward(self, q, k, v, key_padding_mask=None):
        # key_padding_mask: (B, S) com True nas posições a IGNORAR (convenção do
        # nn.MultiheadAttention). Necessário desde que o ramo textual passou a
        # entregar sequências de comprimento variável com padding: sem a máscara
        # o softmax distribui peso sobre posições nulas.
        attn_output, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        attn_output = self.dropout(attn_output)
        gate = torch.sigmoid(self.gate_linear(q))
        out = gate * attn_output + (1 - gate) * q
        return self.norm(out)


class StackedGatedResidualBlock(nn.Module):
    def __init__(self, dim, depth=4, dropout=0.1):
        super(StackedGatedResidualBlock, self).__init__()
        self.blocks = nn.ModuleList([
            GatedAlteredResidualBlock(dim=dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, q, k=None, v=None, key_padding_mask=None):
        """
        Permite q = k = v, mas também aceita entrada personalizada.

        Args:
            q (Tensor): Query, formato (seq_len, batch, dim)
            k (Tensor): Key (opcional, default=q)
            v (Tensor): Value (opcional, default=q)
            key_padding_mask (Tensor): (batch, seq_len), True = ignorar
        """
        if k is None: k = q
        if v is None: v = q

        for block in self.blocks:
            q = block(q, k, v, key_padding_mask=key_padding_mask)
        return q