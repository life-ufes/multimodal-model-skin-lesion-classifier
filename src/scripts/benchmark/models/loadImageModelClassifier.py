import torch
import torch.nn as nn
import timm
from torchvision import models
from transformers import AutoModel

# CORREÇÃO (Parte 12): o import de TabTransformer no topo do módulo derrubava
# TODA a cadeia de imports (train -> multimodalIntraModalWithBert ->
# loadImageModelClassifier) quando `tab_transformer` não estava disponível,
# mesmo em execuções que nunca usam esse encoder. Passou a ser lazy, dentro do
# ramo que realmente o instancia.

# Dimensão dos sentence embeddings do neuml/pubmedbert-base-embeddings.
# Usada apenas como fallback: o valor autoritativo vem do dataset
# (SkinLesionDataset.embedding_dim), lido do próprio SentenceTransformer.
PUBMEDBERT_EMBEDDING_DIM = 768

PUBMEDBERT_ENCODERS = [
    "pubmedbert-base-embeddings",
    "pubmedbert-base-embeddings-100K",
    "pubmedbert-base-embeddings-500K",
    "pubmedbert-base-embeddings-1M",
    "pubmedbert-base-embeddings-2M",
]


class loadModels:

    # ======================================================
    # Utilitário: controle de fine-tuning do backbone
    # ======================================================
    @staticmethod
    def set_backbone_train_mode(model, mode="frozen_weights", last_n_layers=1):
        # Passo 1: Congela todos os parâmetros como base
        for p in model.parameters():
            p.requires_grad = False

        # Passo 2: Aplica a lógica desejada
        if mode == "frozen_weights":
            pass # Já estão congelados

        elif mode == "unfrozen_weights":
            for p in model.parameters():
                p.requires_grad = True

        elif mode == "last_layer_unfrozen_weights":
            # Coleta todos os parâmetros em uma lista para acessar os finais
            params = list(model.parameters())
            # Multiplicamos por 2 como heurística básica para pegar peso e bias da(s) última(s) camada(s)
            for p in params[-last_n_layers * 2:]:
                p.requires_grad = True

        # CORREÇÃO (Parte 10): "partial" só era tratado nos ramos densenet169 e
        # TIMM. Nos backbones torchvision (resnet, vgg, mobilenet, efficientnet)
        # o valor chegava aqui e levantava ValueError. Fallback genérico:
        # descongela o último filho do módulo.
        elif mode == "partial":
            children = list(model.children())
            if len(children) > 0:
                for p in children[-1].parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"Invalid backbone_train_mode: {mode}")

    # ======================================================
    # Utilitário: inferência da dimensão de saída do backbone
    # ======================================================
    @staticmethod
    def infer_cnn_output_dim(model, input_size=(3, 224, 224), device="cpu"):
        # CORREÇÃO (Parte 11): este método era chamado no ramo TIMM mas nunca
        # existiu na classe — qualquer backbone sem `num_features` válido
        # levantava AttributeError em vez de inferir a dimensão.
        was_training = model.training
        model.eval()
        original_device = next(model.parameters()).device

        try:
            model.to(device)
            with torch.no_grad():
                dummy = torch.zeros(1, *input_size, device=device)
                out = model(dummy)

            if isinstance(out, (list, tuple)):
                out = out[0]

            # Backbones que devolvem mapas espaciais: achata tudo além do batch.
            if out.dim() > 2:
                out = torch.flatten(out, start_dim=1)

            return int(out.shape[1])
        finally:
            model.to(original_device)
            if was_training:
                model.train()

    # ======================================================
    # Image encoders
    # ======================================================
    @staticmethod
    def loadModelImageEncoder(
        cnn_model_name: str,
        common_dim: int,
        # CORREÇÃO (Parte 9): o default "frozen" não é um modo válido de
        # set_backbone_train_mode ("frozen_weights"), então qualquer chamada
        # que omitisse o argumento morria com ValueError.
        backbone_train_mode: str = "frozen_weights",
        device: str = "cpu"
    ):
        # ======================================================
        # Custom CNN
        # ======================================================
        if cnn_model_name == "custom-cnn":
            model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, common_dim)
            )
            cnn_dim_output = common_dim
            loadModels.set_backbone_train_mode(model, backbone_train_mode)

        # ======================================================
        # TorchVision
        # ======================================================
        elif cnn_model_name == "resnet-18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Identity()
            cnn_dim_output = 512
            loadModels.set_backbone_train_mode(model, backbone_train_mode, last_n_layers=1)

        elif cnn_model_name == "resnet-50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Identity()
            cnn_dim_output = 2048
            loadModels.set_backbone_train_mode(model, backbone_train_mode, last_n_layers=1)

        elif cnn_model_name == "vgg16":
            model = models.vgg16(pretrained=True)
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            cnn_dim_output = 4096
            loadModels.set_backbone_train_mode(model, backbone_train_mode, last_n_layers=1)

        elif cnn_model_name == "densenet169":
            model = models.densenet169(pretrained=True)
            model.classifier = nn.Identity()
            cnn_dim_output = 1664

            if backbone_train_mode == "partial":
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.features.denseblock4.parameters():
                    p.requires_grad = True
            else:
                loadModels.set_backbone_train_mode(model, backbone_train_mode)

        elif cnn_model_name == "mobilenet-v2":
            model = models.mobilenet_v2(pretrained=True)
            model.classifier = nn.Identity()
            cnn_dim_output = 1280
            loadModels.set_backbone_train_mode(model, backbone_train_mode, last_n_layers=1)

        elif cnn_model_name == "efficientnet-b0":
            model = models.efficientnet_b0(pretrained=True)
            model.classifier = nn.Identity()
            cnn_dim_output = 1280
            loadModels.set_backbone_train_mode(model, backbone_train_mode, last_n_layers=1)

        elif cnn_model_name == "efficientnet-b4":
            model = models.efficientnet_b4(pretrained=True)
            model.classifier = nn.Identity()
            cnn_dim_output = 1792
            loadModels.set_backbone_train_mode(model, backbone_train_mode, last_n_layers=1)

        elif cnn_model_name == "efficientnet-b7":
            model = models.efficientnet_b7(pretrained=True)
            model.classifier = nn.Identity()
            cnn_dim_output = 2560
            loadModels.set_backbone_train_mode(model, backbone_train_mode, last_n_layers=1)

        # ======================================================
        # TIMM genérico
        # ======================================================
        elif cnn_model_name in timm.list_models(pretrained=True):
            model = timm.create_model(cnn_model_name, pretrained=True)

            # reset_classifier(0) preserva o global_pool: a saída é o vetor
            # pooled (B, D). Mapas espaciais exigiriam reset_classifier(0, "").
            if hasattr(model, "reset_classifier"):
                model.reset_classifier(0)

            if backbone_train_mode == "partial":
                for p in model.parameters():
                    p.requires_grad = False

                if hasattr(model, "stages"):
                    for p in model.stages[-1].parameters():
                        p.requires_grad = True
                elif hasattr(model, "blocks"):
                    # pode ser ModuleList, então libera o último bloco
                    last_block = model.blocks[-1]
                    for p in last_block.parameters():
                        p.requires_grad = True
                else:
                    # fallback: último child
                    children = list(model.children())
                    if len(children) > 0:
                        for p in children[-1].parameters():
                            p.requires_grad = True

            else:
                loadModels.set_backbone_train_mode(model, backbone_train_mode)

            # tenta usar num_features; se não existir ou falhar, infere
            cnn_dim_output = getattr(model, "num_features", None)
            if cnn_dim_output is None or cnn_dim_output <= 0:
                cnn_dim_output = loadModels.infer_cnn_output_dim(
                    model=model,
                    input_size=(3, 224, 224),
                    device=device
                )

        else:
            raise ValueError(f"Backbone '{cnn_model_name}' não implementado.")

        return model, cnn_dim_output

    # ======================================================
    # Text encoders
    # ======================================================
    @staticmethod
    def loadTextModelEncoder(
        text_model_encoder: str,
        train_mode: str = "frozen_weights"
    ):

        # ------------------------------
        # HuggingFace Transformers
        # ------------------------------
        if text_model_encoder in ["bert-base-uncased", "gpt2"]:
            model = AutoModel.from_pretrained(text_model_encoder)
            output_dim = model.config.hidden_size

            if train_mode == "unfrozen_weights":
                for p in model.parameters():
                    p.requires_grad = True
            else:
                for p in model.parameters():
                    p.requires_grad = False

            return model, output_dim, output_dim

        # --- PubMedBERT (sentence embeddings pré-computados) ---
        elif text_model_encoder in PUBMEDBERT_ENCODERS:
            # Não há encoder residente: o dataset entrega o vetor já calculado.
            # CORREÇÃO (Parte 13): o 64 hardcoded não correspondia a nenhum
            # modelo em uso — neuml/pubmedbert-base-embeddings produz 768.
            # MultimodalModel ignora este valor (usa embedding_dim do dataset)
            # quando o encoder é None; o retorno abaixo serve aos demais
            # chamadores, que não têm essa informação.
            output_dim = PUBMEDBERT_EMBEDDING_DIM
            return None, output_dim, output_dim

        # ------------------------------
        # TabTransformer
        # ------------------------------
        elif text_model_encoder == "tab-transformer":
            from tab_transformer import TabTransformer

            # ATENÇÃO: range(82) criaria cardinalidades erradas ([0, 1, 2...]).
            # Crie uma lista com o número real de classes para cada categoria.
            # Exemplo de fallback seguro (assumindo que as 82 colunas tenham max 10 opções cada):
            # O ideal é injetar a lista real de cardinalidades baseada no dataset.
            categorical_cardinalities = [10] * 82
            output_dim = 85

            model = TabTransformer(
                categorical_cardinalities=categorical_cardinalities,
                num_continuous=4,
                output_dim=output_dim
            )

            return model, output_dim, output_dim

        else:
            raise ValueError(f"Text encoder '{text_model_encoder}' não suportado.")