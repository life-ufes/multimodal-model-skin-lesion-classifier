import torch
import torch.nn as nn
import timm
from torchvision import models
from tab_transformer import TabTransformer
from transformers import ViTModel, CLIPModel, CLIPProcessor, AutoModel, BertModel
class loadModels():
    @staticmethod
    def loadModelImageEncoder(cnn_model_name, common_dim, unfreeze_weights=False):
        ''' Seleciona o modelo desejado e entrega-o mesmo assim como as dimensões da sua saída '''
        try:
            if cnn_model_name == "custom-cnn":
                image_encoder = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Flatten(),
                    nn.Linear(16 * 56 * 56, common_dim)
                )
                cnn_dim_output = common_dim

            elif cnn_model_name == "resnet-50":
                image_encoder = models.resnet50(pretrained=True)
                cnn_dim_output = 2048

                # Congelar os pesos da ResNet-50
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
                # Substituir a camada final por uma identidade
                image_encoder.fc = nn.Identity()

            elif cnn_model_name == "resnet-18":
                image_encoder = models.resnet18(pretrained=True)
                cnn_dim_output = 512
                # Congelar os pesos da ResNet-18
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
                # Substituir a camada final por uma identidade
                image_encoder.fc = nn.Identity()

            elif cnn_model_name == "vgg16":
                image_encoder = models.vgg16(pretrained=True)
                cnn_dim_output = 4096
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights

                # Ajustar a saída para manter a dimensão esperada (4096)
                image_encoder.classifier = nn.Sequential(
                    *list(image_encoder.classifier.children())[:-1],  # Remover a última camada (1000 classes)
                    nn.Linear(4096, 4096)  # Garantir que a saída permanece 4096
                )

            elif cnn_model_name == "densenet169":
                image_encoder = models.densenet169(pretrained=True)
                cnn_dim_output = 1664
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
                # # Unfreeze some layers
                # for param in list(image_encoder.features[-1:].parameters()):
                #     param.requires_grad = True
                # Ajustar a saída para manter a dimensão esperada (4096)
                image_encoder.classifier = nn.Sequential(
                    *list(image_encoder.classifier.children())[:-1],  # Remover a última camada (1000 classes)
                    nn.Linear(1664, 1664)  # Garantir que a saída permanece 4096
                )

            elif cnn_model_name == "mobilenet-v2":
                image_encoder = models.mobilenet_v2(pretrained=True)
                cnn_dim_output = 1280
                # Congelar os pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights

                # Ajustar a saída para manter a dimensão esperada (1280)
                image_encoder.classifier = nn.Sequential(
                    *list(image_encoder.classifier.children())[:-1],  # Remover a última camada (1000 classes)
                    nn.Linear(cnn_dim_output, cnn_dim_output)  # Manter a dimensão
                )
            elif cnn_model_name == "google/vit-base-patch16-224":
                # Carregar o modelo ViT pré-treinado
                image_encoder = ViTModel.from_pretrained(f"google/{cnn_model_name}")
                cnn_dim_output = image_encoder.config.hidden_size  # Ajustando a saída conforme o ViT
                # Congelar os pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights


            elif cnn_model_name == "openai/clip-vit-base-patch16":
                # Load the CLIP model and extract the vision encoder
                clip_model = CLIPModel.from_pretrained(f"{cnn_model_name}")
                image_encoder = clip_model.vision_model  # Extract only the vision encoder

                # Get the output dimension of the vision model
                cnn_dim_output = clip_model.config.vision_config.hidden_size

                # Freeze weights if necessary
                if not unfreeze_weights:
                    for param in image_encoder.parameters():
                        param.requires_grad = unfreeze_weights

            elif cnn_model_name == "dinov2_vits14":  # Escolha a variante correta
                # Carregar o modelo DINOv2 usando torch.hub
                image_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

                # Ajustar a saída do modelo
                cnn_dim_output = image_encoder.embed_dim  # O DINOv2 não usa config.hidden_size como o ViT

                # Congelar os pesos se necessário
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
        
            elif cnn_model_name == "nextvit_small.bd_ssld_6m_in1k":
                image_encoder = timm.create_model(cnn_model_name, pretrained=True)

                # Ajusta o input_size padrão para 224x224
                image_encoder.default_cfg['input_size'] = (3, 224, 224)

                # Remove a última camada de classificação
                image_encoder.reset_classifier(0)

                # A dimensão de saída agora é a do penúltimo bloco (features)
                cnn_dim_output = image_encoder.num_features  # ou usar diretamente com forward_features()

                # Congelar ou descongelar pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
            elif cnn_model_name == "caformer_b36.sail_in22k_ft_in1k":
                image_encoder = timm.create_model(cnn_model_name, pretrained=True)

                # Ajusta o input_size padrão para 224x224
                image_encoder.default_cfg['input_size'] = (3, 224, 224)

                # Remove a última camada de classificação
                image_encoder.reset_classifier(0)

                # A dimensão de saída agora é a do penúltimo bloco (features)
                cnn_dim_output = image_encoder.num_features  # ou usar diretamente com forward_features()

                # Congelar ou descongelar pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
            
            elif cnn_model_name == "beit_large_patch16_224.in22k_ft_in22k_in1k":
                image_encoder = timm.create_model(cnn_model_name, pretrained=True)

                # Ajusta o input_size padrão para 224x224
                image_encoder.default_cfg['input_size'] = (3, 224, 224)

                # Remove a última camada de classificação
                image_encoder.reset_classifier(0)

                # A dimensão de saída agora é a do penúltimo bloco (features)
                cnn_dim_output = image_encoder.num_features  # ou usar diretamente com forward_features()

                # Congelar ou descongelar pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
            
            elif cnn_model_name == "beitv2_large_patch16_224.in1k_ft_in22k_in1k":
                image_encoder = timm.create_model(cnn_model_name, pretrained=True)

                # Ajusta o input_size padrão para 224x224
                image_encoder.default_cfg['input_size'] = (3, 224, 224)

                # Remove a última camada de classificação
                image_encoder.reset_classifier(0)

                # A dimensão de saída agora é a do penúltimo bloco (features)
                cnn_dim_output = image_encoder.num_features  # ou usar diretamente com forward_features()

                # Congelar ou descongelar pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
            
            elif cnn_model_name == "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k":
                image_encoder = timm.create_model(cnn_model_name, pretrained=True)

                # Ajusta o input_size padrão para 224x224
                image_encoder.default_cfg['input_size'] = (3, 224, 224)

                # Remove a última camada de classificação
                image_encoder.reset_classifier(0)

                # A dimensão de saída agora é a do penúltimo bloco (features)
                cnn_dim_output = image_encoder.num_features  # ou usar diretamente com forward_features()

                # Congelar ou descongelar pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
            
            elif cnn_model_name == "mvitv2_small.fb_in1k":
                image_encoder = timm.create_model(cnn_model_name, pretrained=True)

                # Ajusta o input_size padrão para 224x224
                image_encoder.default_cfg['input_size'] = (3, 224, 224)

                # Remove a última camada de classificação
                image_encoder.reset_classifier(0)

                # A dimensão de saída agora é a do penúltimo bloco (features)
                cnn_dim_output = image_encoder.num_features  # ou usar diretamente com forward_features()

                # Congelar ou descongelar pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
            elif cnn_model_name == "coat_lite_small.in1k":
                image_encoder = timm.create_model(cnn_model_name, pretrained=True)

                # Ajusta o input_size padrão para 224x224
                image_encoder.default_cfg['input_size'] = (3, 224, 224)

                # Remove a última camada de classificação
                image_encoder.reset_classifier(0)

                # A dimensão de saída agora é a do penúltimo bloco (features)
                cnn_dim_output = image_encoder.num_features  # ou usar diretamente com forward_features()

                # Congelar ou descongelar pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
            elif cnn_model_name == "davit_tiny.msft_in1k":
                image_encoder = timm.create_model(cnn_model_name, pretrained=True)

                # Ajusta o input_size padrão para 224x224
                image_encoder.default_cfg['input_size'] = (3, 224, 224)

                # Remove a última camada de classificação
                image_encoder.reset_classifier(0)

                # A dimensão de saída agora é a do penúltimo bloco (features)
                cnn_dim_output = image_encoder.num_features  # ou usar diretamente com forward_features()

                # Congelar ou descongelar pesos
                for param in image_encoder.parameters():
                    param.requires_grad = unfreeze_weights
            
            else:
                raise ValueError("CNN não implementada.")
            return image_encoder, cnn_dim_output

        except Exception as e:
            print(f"Erro ao tentar carregar o modelo!. Erro: {e}\n")

    @staticmethod
    def loadTextModelEncoder(text_model_encoder:str, unfreeze_weights: bool=False):
        if text_model_encoder == "bert-base-uncased":
            # Definir encoder de texto (BERT)
            text_encoder = AutoModel.from_pretrained(text_model_encoder)
            for param in text_encoder.parameters():
                param.requires_grad = unfreeze_weights

            bert_output_dim = text_encoder.config.hidden_size
            vocab_size = 768
            return text_encoder, bert_output_dim, vocab_size
        
        # --- GPT-2 ---
        elif text_model_encoder.startswith("gpt2"):
            text_encoder = AutoModel.from_pretrained(text_model_encoder)
            # Congela todos os pesos do GPT-2
            for param in text_encoder.parameters():
                param.requires_grad = unfreeze_weights
            output_dim = text_encoder.config.hidden_size
            vocab_size = output_dim
            return text_encoder, output_dim, vocab_size

        elif text_model_encoder == "tab-transformer":
            # Defina as cardinalidades reais para suas colunas categóricas
            # Exemplo: se você tiver 4 colunas com 10, 15, 20 e 25 categorias:
            categorical_indices = list(range(82))  # Vetor [0, 1, 2, ..., 81]
            vocab_size=85
            text_encoder_model_output=85 # 85
            text_encoder = TabTransformer(
                categorical_cardinalities=categorical_indices,  # Cardinalidades das features categóricas
                num_continuous=4,  # Número de colunas contínuas
                output_dim=vocab_size   # Ajustando a dimensão de saída para 512
            )
            return text_encoder, text_encoder_model_output, vocab_size
        else:
            return None, None, None
