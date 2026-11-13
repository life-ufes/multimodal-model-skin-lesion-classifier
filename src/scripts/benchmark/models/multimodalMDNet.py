import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from loadImageModelClassifier import loadModels

# Corrected MetaNet module
class MetaNet(nn.Module):
    """
    Implementação da abordagem MetaNet
    Fusing Metadata and Dermoscopy Images for Skin Disease Diagnosis - https://ieeexplore.ieee.org/document/9098645
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(MetaNet, self).__init__()
        self.metanet = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 1),
            nn.ReLU(),
            nn.Conv2d(middle_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_maps, metadata):
        # metadata: tensor de dimensão [B, in_channels]
        # Adiciona dimensões espaciais para compatibilidade com Conv2d: [B, in_channels, 1, 1]
        metadata = metadata.unsqueeze(-1).unsqueeze(-1)
        # Passa os metadados pela sequência de convoluções para gerar um mapa de pesos
        x = self.metanet(metadata)
        # Multiplica o mapa de pesos com os mapas de features extraídos da imagem
        x = x * feat_maps
        return x

# Corrected MetaBlock module
class MetaBlock(nn.Module):
    """
    Implementação do Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U):
        """
        V: número de canais de features da imagem (ex.: 1664 da DenseNet-169)
        U: dimensão dos metadados (ex.: 85)
        """
        super(MetaBlock, self).__init__()
        self.fb = nn.Sequential(nn.Linear(U, V), nn.LayerNorm(V))
        self.gb = nn.Sequential(nn.Linear(U, V), nn.LayerNorm(V))

    def forward(self, img_features, metadata):
        # img_features: tensor de features da imagem com forma [B, V, H, W]
        # metadata: tensor de metadados com forma [B, U]
        t1 = self.fb(metadata)  # [B, V]
        t2 = self.gb(metadata)  # [B, V]
        # Expandir dimensões para compatibilidade com img_features (assumindo [B, V, H, W])
        t1 = t1.unsqueeze(-1).unsqueeze(-1)  # [B, V, 1, 1]
        t2 = t2.unsqueeze(-1).unsqueeze(-1)  # [B, V, 1, 1]
        # Modulação das features: aplica tanh na multiplicação e soma t2, seguido de sigmoid
        out = torch.sigmoid(torch.tanh(img_features * t1) + t2)
        return out


# ======================================================
# Wrapper TIMM: expõe os mapas espaciais (forward_features)
# ======================================================
class _TimmSpatial(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model.forward_features(x)
        if out.dim() != 4:
            raise RuntimeError(
                f"O backbone TIMM devolveu shape {tuple(out.shape)}; o MD-Net exige "
                "mapas espaciais [B, C, H, W] (backbones tipo ViT não são suportados)."
            )
        return out


# Corrected MD-Net module
class MDNet(nn.Module):
    def __init__(self,
            meta_dim=85,
            num_classes=6,
            cnn_model_name="densenet169",
            text_model_name="one-hot-encode",
            hidden_dim=128,
            device="cpu",
            unfreeze_weights: str="partial",
            last_n_layers: int=1,
            input_size=(3, 224, 224)
        ):
        super(MDNet, self).__init__()
        self.device = device
        self.meta_dim = meta_dim
        self.num_classes = num_classes
        self.cnn_model_name = cnn_model_name
        self.text_model_name = text_model_name
        self.unfreeze_weights = unfreeze_weights
        # CORREÇÃO: common_dim não existia. Ele só é usado pela custom-cnn
        # (dimensão da Linear final, descartada abaixo), então usamos hidden_dim.
        self.common_dim = hidden_dim

        # 1. Carrega o backbone completo. O modo de treino é aplicado depois,
        #    no extrator espacial (a parte que o MD-Net de fato usa).
        backbone, _ = loadModels.loadModelImageEncoder(
            cnn_model_name=self.cnn_model_name,
            common_dim=self.common_dim,
            backbone_train_mode="frozen_weights",
            device=device
        )

        # 2. CORREÇÃO: loadModelImageEncoder devolve o vetor já pooled [B, D]
        #    (fc/classifier = Identity, mas o avgpool continua). MetaNet e
        #    MetaBlock precisam dos mapas espaciais [B, C, H', W'].
        self.feature_extractor = self._build_spatial_extractor(backbone)

        # 3. CORREÇÃO: num_channels era fixo em 1664 (DenseNet-169).
        #    Para resnet-50 são 2048; vgg16 dá 512 (não 4096). Inferimos.
        self.num_channels = self._infer_num_channels(input_size)

        # 4. Aplica o modo de fine-tuning ao extrator espacial
        self.set_backbone_train_mode(self.unfreeze_weights, last_n_layers=last_n_layers)

        # Módulo MetaNet: os metadados de dimensão meta_dim serão transformados em um mapa de pesos com num_channels
        self.meta_net = MetaNet(in_channels=meta_dim, middle_channels=hidden_dim, out_channels=self.num_channels)
        # Módulo MetaBlock: V = número de canais da imagem e U = dimensão dos metadados
        self.meta_block = MetaBlock(V=self.num_channels, U=meta_dim)

        # Classificador final
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.num_channels, self.num_classes)

    # ======================================================
    # Extrator espacial por arquitetura
    # ======================================================
    def _build_spatial_extractor(self, backbone):
        name = self.cnn_model_name

        if name == "densenet169":
            # DenseNet.forward aplica ReLU após `features`; mantido por equivalência
            return nn.Sequential(backbone.features, nn.ReLU(inplace=False))

        if name in ("resnet-18", "resnet-50"):
            # remove avgpool e fc -> [B, 512/2048, H/32, W/32]
            return nn.Sequential(*list(backbone.children())[:-2])

        if name in ("vgg16", "mobilenet-v2", "efficientnet-b0", "efficientnet-b4", "efficientnet-b7"):
            return backbone.features

        if name == "custom-cnn":
            # conv -> relu -> maxpool (descarta pool global, flatten e linear)
            return nn.Sequential(*list(backbone.children())[:3])

        if hasattr(backbone, "forward_features"):  # TIMM
            return _TimmSpatial(backbone)

        raise ValueError(f"Backbone '{name}' sem extrator espacial definido no MD-Net.")

    @torch.no_grad()
    def _infer_num_channels(self, input_size):
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        p = next(self.feature_extractor.parameters())
        out = self.feature_extractor(torch.zeros(1, *input_size, device=p.device, dtype=p.dtype))
        self.feature_extractor.train(was_training)
        return int(out.shape[1])

    # ======================================================
    # Integração com loadModels.set_backbone_train_mode
    # ======================================================
    def set_backbone_train_mode(self, mode: str, last_n_layers: int = 1):
        if mode == "partial":
            # O fallback genérico de "partial" descongela o último filho do módulo,
            # que aqui pode não ter parâmetros (ex.: ReLU da DenseNet). Por isso o
            # bloco a liberar é escolhido por arquitetura.
            loadModels.set_backbone_train_mode(self.feature_extractor, "frozen_weights")
            for p in self._partial_block().parameters():
                p.requires_grad = True
        else:
            loadModels.set_backbone_train_mode(
                self.feature_extractor, mode, last_n_layers=last_n_layers
            )
        self.unfreeze_weights = mode

    def _partial_block(self):
        fe = self.feature_extractor
        if self.cnn_model_name == "densenet169":
            return fe[0].denseblock4
        if isinstance(fe, _TimmSpatial):
            m = fe.model
            if hasattr(m, "stages"):
                return m.stages[-1]
            if hasattr(m, "blocks"):
                return m.blocks[-1]
            if hasattr(m, "layer4"):
                return m.layer4
            fe = m
        # resnet -> layer4; vgg/mobilenet/efficientnet -> última camada de features
        for child in reversed(list(fe.children())):
            if any(True for _ in child.parameters()):
                return child
        return fe

    def forward(self, image, metadata):
        # image: [B, 3, H, W]
        # metadata: [B, meta_dim]
        # 1. Extrai os mapas de features visuais do backbone
        image_features = self.feature_extractor(image)  # [B, num_channels, H', W']

        # 2. Aplica o MetaNet para obter pesos canal a canal
        meta_net_features = self.meta_net(image_features, metadata)  # [B, num_channels, H', W']

        # 3. Aplica o MetaBlock para refinar as features com atenção orientada por metadados
        meta_block_features = self.meta_block(image_features, metadata)  # [B, num_channels, H', W']

        # 4. Fusão dos outputs (soma element-wise)
        fused_features = meta_net_features + meta_block_features

        # 5. Pooling global e classificação
        pooled_features = self.avg_pool(fused_features)  # [B, num_channels, 1, 1]
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # [B, num_channels]
        out = self.classifier(pooled_features)  # [B, num_classes]
        return out