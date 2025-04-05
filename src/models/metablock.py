import torch
from torch import nn

class MetaBlock(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def init(self, V, U):
        super(MetaBlock, self).init()
        # V is the number of image feature channels (e.g., 1664 from DenseNet-169)
        # U is the metadata dimension (e.g., 85 in your case)
        self.fb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V, U):
        # V: image features, expected shape [B, C, ] where C == V (e.g., [B, 1664, H, W])
        # U: metadata vector, shape [B, U] (e.g., [B, 85])
        t1 = self.fb(U)  # produces a tensor of shape [B, V]
        t2 = self.gb(U)  # produces a tensor of shape [B, V]
        V = torch.sigmoid(torch.tanh(V* t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V
