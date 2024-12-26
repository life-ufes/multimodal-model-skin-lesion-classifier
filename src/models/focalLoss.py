import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implementação básica da Focal Loss para classificação multi-classe.
    É possível extender para casos binários ou multi-label, mas aqui focamos
    na tipologia multi-class (similar a CrossEntropyLoss).
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        alpha (tensor): pesos para cada classe (similar a class_weights da CE).
        gamma (int ou float): fator de foco. Típicos: 1, 2, 3.
        reduction (str): 'mean', 'sum' ou 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits de dimensões (batch_size, num_classes)
        targets: rótulos de dimensões (batch_size) - classes numéricas
        """
        # Primeiro calculamos a CrossEntropy "padrão" (sem softmax explícito)
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.alpha, 
            reduction='none'
        )
        # pt é a probabilidade predita correta: exp(-CE)
        pt = torch.exp(-ce_loss)
        # Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Redução final
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # sem redução
