import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel
from transformers import AutoTokenizer, AutoModel

class loadModels():
    def loadModelImageEncoder(cnn_model_name, common_dim):
        ''' Seleciona o modelo desejado e entrega-o mesmo assim como as dimensões da sua saída'''
        try:
            if cnn_model_name == "custom-cnn":
                image_encoder = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Flatten(),
                    nn.Linear(16 * 56 * 56, common_dim)
                )
            elif cnn_model_name == "resnet-50":
                image_encoder = models.resnet50(pretrained=True)
                cnn_dim_output = 2048
                # Congelar os pesos da ResNet-50
                for param in image_encoder.parameters():
                    param.requires_grad = False
                # Substituir a camada final por uma identidade
                image_encoder.fc = nn.Identity()
            elif cnn_model_name == "resnet-18":
                image_encoder = models.resnet18(pretrained=True)
                cnn_dim_output = 512
                # Congelar os pesos da ResNet-18
                for param in image_encoder.parameters():
                    param.requires_grad = False
                # Substituir a camada final por uma identidade
                image_encoder.fc = nn.Identity()

            elif cnn_model_name == "vgg16":
                image_encoder = models.vgg16(pretrained=True)
                cnn_dim_output = 4096
                for param in image_encoder.parameters():
                    param.requires_grad = False
                # Ajustar a saída para manter a dimensão esperada (4096)
                image_encoder.classifier = nn.Sequential(
                    *list(image_encoder.classifier.children())[:-1],  # Remover a última camada (1000 classes)
                    nn.Linear(4096, 4096)  # Garantir que a saída permanece 4096
                )

            elif cnn_model_name == "mobilenet-v2":
                image_encoder = models.mobilenet_v2(pretrained=True)
                cnn_dim_output = 1280
                    # Congelar os pesos
                for param in image_encoder.parameters():
                    param.requires_grad = False
                # Ajustar a saída para manter a dimensão esperada (4096)
                image_encoder.classifier = nn.Sequential(
                    *list(image_encoder.classifier.children())[:-1],  # Remover a última camada (1000 classes)
                    nn.Linear(cnn_dim_output, cnn_dim_output)  # Garantir que a saída permanece 4096
                )
            elif cnn_model_name == "vit-base-patch16-224":
                # Carregar o modelo ViT pré-treinado
                image_encoder = ViTModel.from_pretrained(f"google/{cnn_model_name}")
                cnn_dim_output = image_encoder.config.hidden_size  # Ajustando a saída conforme o ViT

            else:
                raise ValueError("CNN não implementada.")
            return image_encoder, cnn_dim_output
        
        except Exception as e:
            print(f"Erro ao tentar carregar o modelo!. Erro: {e}\n")

    def loadTextModelEncoder(text_model_encoder):
        bert_model = AutoModel.from_pretrained(text_model_encoder)
        # Congelar os pesos
        for param in bert_model.parameters():
            param.requires_grad = False
        # Dimensão da saída do 
        text_encoder_dim_output = 1024
        return bert_model, text_encoder_dim_output