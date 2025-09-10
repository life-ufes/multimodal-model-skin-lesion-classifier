import torch
import torch.nn  as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super(SoftTargetCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        """
        inputs: logits do modelo (batch_size, num_classes)
        targets: soft labels (batch_size, num_classes)
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        if self.weight is not None:
            # aplica pesos por classe
            weight = self.weight.unsqueeze(0)  # (1, num_classes)
            loss = -(targets * log_probs * weight).sum(dim=-1).mean()
        else:
            loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss
