import torch
import torch.nn as nn
import timm
from torchvision import models
from transformers import AutoModel
from tab_transformer import TabTransformer


class loadModels:

    # ======================================================
    # Utilitário: controle de fine-tuning do backbone
    # ======================================================
    @staticmethod
    def set_backbone_train_mode(model, mode="frozen_weights", last_n_layers=1):
        if mode == "frozen_weights":
            for p in model.parameters():
                p.requires_grad = False

        elif mode == "unfrozen_weights":
            for p in model.parameters():
                p.requires_grad = True

        elif mode == "last_layer_unfrozen_weights":
            for p in model.parameters():
                p.requires_grad = False

            children = list(model.children())
            for layer in children[-last_n_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"Invalid backbone_train_mode: {mode}")

    # ======================================================
    # Image encoders
    # ======================================================
    @staticmethod
    def loadModelImageEncoder(
        cnn_model_name: str,
        backbone_train_mode: str = "frozen_weights"
    ):

        # ------------------------------
        # TorchVision CNNs
        # ------------------------------
        if cnn_model_name == "resnet-18":
            model = models.resnet18(pretrained=True)
            cnn_dim = 512
            model.fc = nn.Identity()

            loadModels.set_backbone_train_mode(
                model, backbone_train_mode, last_n_layers=1
            )

        elif cnn_model_name == "resnet-50":
            model = models.resnet50(pretrained=True)
            cnn_dim = 2048
            model.fc = nn.Identity()

            loadModels.set_backbone_train_mode(
                model, backbone_train_mode, last_n_layers=1
            )

        elif cnn_model_name == "densenet169":
            model = models.densenet169(pretrained=True)
            cnn_dim = 1664
            model.classifier = nn.Identity()

            if backbone_train_mode == "last_layer_unfrozen_weights":
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.features.denseblock4.parameters():
                    p.requires_grad = True
            else:
                loadModels.set_backbone_train_mode(model, backbone_train_mode)

        elif cnn_model_name == "mobilenet-v2":
            model = models.mobilenet_v2(pretrained=True)
            cnn_dim = 1280
            model.classifier = nn.Identity()

            loadModels.set_backbone_train_mode(
                model, backbone_train_mode, last_n_layers=1
            )

        elif cnn_model_name == "efficientnet-b0":
            model = models.efficientnet_b0(pretrained=True)
            cnn_dim = 1280
            model.classifier = nn.Identity()

            loadModels.set_backbone_train_mode(
                model, backbone_train_mode, last_n_layers=1
        )

        elif cnn_model_name == "efficientnet-b4":
            model = models.efficientnet_b4(pretrained=True)
            cnn_dim = 1792
            model.classifier = nn.Identity()

            loadModels.set_backbone_train_mode(
                model, backbone_train_mode, last_n_layers=1
        )

        elif cnn_model_name == "efficientnet-b7":
            model = models.efficientnet_b7(pretrained=True)
            cnn_dim = 2560
            model.classifier = nn.Identity()

            loadModels.set_backbone_train_mode(
                model, backbone_train_mode, last_n_layers=1
            )

        # ------------------------------
        # timm models (ViT / Hybrid)
        # ------------------------------
        elif cnn_model_name in timm.list_models(pretrained=True):
            model = timm.create_model(cnn_model_name, pretrained=True)
            model.reset_classifier(0)
            cnn_dim = model.num_features

            if backbone_train_mode == "last_layer_unfrozen_weights":
                for p in model.parameters():
                    p.requires_grad = False

                # estratégia genérica: último estágio
                if hasattr(model, "stages"):
                    for p in model.stages[-1].parameters():
                        p.requires_grad = True
                elif hasattr(model, "blocks"):
                    for p in model.blocks[-1].parameters():
                        p.requires_grad = True
            else:
                loadModels.set_backbone_train_mode(model, backbone_train_mode)

        else:
            raise ValueError(f"Backbone '{cnn_model_name}' não implementado.")

        return model, cnn_dim

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

        # ------------------------------
        # TabTransformer
        # ------------------------------
        elif text_model_encoder == "tab-transformer":
            categorical_indices = list(range(82))
            output_dim = 85

            model = TabTransformer(
                categorical_cardinalities=categorical_indices,
                num_continuous=4,
                output_dim=output_dim
            )

            return model, output_dim, output_dim

        else:
            raise ValueError(f"Text encoder '{text_model_encoder}' não suportado.")