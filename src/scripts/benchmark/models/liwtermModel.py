import torch
import torch.nn as nn
from torchvision import models
from loadImageModelClassifier import loadModels


class LiwTERM(nn.Module):
    """
    Implementação fiel ao paper LiwTERM (SIBGRAPI 2024)
    """

    def __init__(
        self,
        num_classes: int,
        meta_dim: int,
        image_encoder: str = "vit_large_patch16_224",
        unfreeze_weights: str = "partial",
        pretrained: bool = True,          # mantido por compatibilidade; o loadModels sempre carrega pesos pré-treinados
        unfreeze_backbone: bool = False,  # legado: True equivale a unfreeze_weights="unfrozen_weights"
        dropout: float = 0.3,
        last_n_layers: int = 1,
        device: str = "cpu",
    ):
        super().__init__()

        # CORREÇÃO: cnn_model_name, common_dim e unfreeze_weights eram lidos
        # de self.* sem nunca terem sido definidos.
        self.cnn_model_name = image_encoder
        self.common_dim = 4096  # só é usado pela custom-cnn
        self.unfreeze_weights = "unfrozen_weights" if unfreeze_backbone else unfreeze_weights

        # =====================================================
        # 1) Image Encoder — FEATURE EXTRACTOR
        # =====================================================
        # O loadModels já devolve o vetor pooled [B, D]:
        #   - TIMM ViT: reset_classifier(0) mantém global_pool="token" -> CLS token
        #   - TorchVision: fc/classifier = Identity após o pooling
        # O modo de treino é aplicado depois, em set_backbone_train_mode.
        self.image_encoder, self.cnn_dim_output = loadModels.loadModelImageEncoder(
            cnn_model_name=self.cnn_model_name,
            common_dim=self.common_dim,
            backbone_train_mode="frozen_weights",
            device=device,
        )
        # CORREÇÃO: `.features` não existe nos modelos TIMM (ViT) e `self.visual`
        # nunca foi definido. O encoder inteiro é o extrator de features.
        self.visual_dim = self.cnn_dim_output

        self.set_backbone_train_mode(self.unfreeze_weights, last_n_layers=last_n_layers)

        # Projeção ViT → 4096 (como no paper)
        self.visual_proj = nn.Sequential(
            nn.Linear(self.visual_dim, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # =====================================================
        # 2) Projeção dos metadados (OHE)
        # =====================================================
        self.meta_fc = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, 1024),
            nn.ReLU(inplace=True),
        )

        # =====================================================
        # 3) Shallow Lightweight Model (SLM)
        # =====================================================
        concat_dim = 4096 + 1024

        self.slm = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(concat_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes)
        )

    # =====================================================
    # Integração com loadModels.set_backbone_train_mode
    # =====================================================
    def set_backbone_train_mode(self, mode: str, last_n_layers: int = 1):
        if mode == "partial":
            # O fallback genérico de "partial" libera o último filho do módulo,
            # que em resnet/mobilenet/efficientnet é o Identity (sem pesos).
            # Por isso o bloco é escolhido por arquitetura.
            loadModels.set_backbone_train_mode(self.image_encoder, "frozen_weights")
            for p in self._partial_block().parameters():
                p.requires_grad = True
        else:
            loadModels.set_backbone_train_mode(
                self.image_encoder, mode, last_n_layers=last_n_layers
            )
        self.unfreeze_weights = mode

    def _partial_block(self):
        m = self.image_encoder
        if self.cnn_model_name == "densenet169":
            return m.features.denseblock4
        if hasattr(m, "stages"):   # TIMM ConvNeXt, etc.
            return m.stages[-1]
        if hasattr(m, "blocks"):   # TIMM ViT / EfficientNet
            return m.blocks[-1]
        if hasattr(m, "layer4"):   # ResNets
            return m.layer4

        def last_with_params(module):
            for name, child in reversed(list(module.named_children())):
                if any(True for _ in child.parameters()):
                    return name, child
            return None, module

        # vgg -> classifier (Linear 4096); mobilenet/efficientnet -> features[-1]
        name, block = last_with_params(m)
        if name == "features":
            _, block = last_with_params(block)
        return block

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:

        # --- Visual features (CLS token no ViT / vetor pooled nas CNNs) ---
        v = self.image_encoder(image)  # [B, visual_dim]
        if v.dim() > 2:
            v = torch.flatten(v, start_dim=1)

        v = self.visual_proj(v)

        # --- Metadata features ---
        m = self.meta_fc(metadata)

        # --- Concatenação ---
        x = torch.cat([v, m], dim=1)

        # --- Classificação ---
        logits = self.slm(x)
        return logits