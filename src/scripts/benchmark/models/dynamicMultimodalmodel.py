import os
import torch
import torch.nn as nn
import sys
# Adicione sys.path se necessário, mas geralmente é melhor configurar o PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from gatedResidualBlock import GatedAlteredResidualBlock
from metablock import MetaBlock

class DynamicCNN(nn.Module):
    def __init__(self, config: dict, in_channels: int = 3, num_classes: int = 10, attention_mecanism: str = "concatenation",
                 text_model_name="one-hot-encoder", n: int = 2, device="cpu", num_heads: int = 8, common_dim: int = 512, vocab_size: int = 85, text_encoder_dim_output:int=512):
        super().__init__()
        self.config = config
        self.device = device
        self.common_dim = common_dim
        self.text_encoder_dim_output = text_encoder_dim_output
        self.num_heads = num_heads
        self.n = n
        self.attention_mecanism = attention_mecanism
        self.text_model_name = text_model_name
        self.vocab_size = vocab_size
        self.k = int(config.get("kernel_size", 3))
        self.hidden_dim = int(config.get("mlp_hidden_dim", 512))
        self.num_classes = int(num_classes)
        self.num_layers_text_fc = int(config.get("num_layers_text_fc", 2))
        self.neurons_per_layer_size_of_text_fc = int(config.get("neurons_per_layer_size_of_text_fc", 512))
        self.num_layers_fc_module = int(config.get("num_layers_fc_module", 2))
        self.neurons_per_layer_size_of_fc_module = int(config.get("neurons_per_layer_size_of_fc_module", 1024))

        # CNN Backbone (otimizável)
        self.layers = []
        self.in_channels = in_channels
        filters = config.get("filters", [64, 128, 256])  # lista controlada por NNI
        self.out_channels = filters[0]

        for out_channels in filters:
            for _ in range(int(config.get("layers_per_block", 2))):
                self.layers.append(nn.Conv2d(int(self.in_channels), int(out_channels), kernel_size=int(self.k), padding=int(self.k) // 2, bias=False))
                self.layers.append(nn.BatchNorm2d(int(out_channels)))
                self.layers.append(nn.ReLU(inplace=True))
                self.in_channels = int(out_channels)
            if config.get("use_pooling", True):
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block = nn.Sequential(*self.layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_encoder = nn.Sequential(self.conv_block, self.global_pool)
        self.cnn_dim_output = filters[-1]

        # Text Encoder
        neurons_per_layer_size_of_text_fc = []
        neurons_per_layer_size_of_text_fc.append(nn.Linear(vocab_size, self.neurons_per_layer_size_of_text_fc))
        neurons_per_layer_size_of_text_fc.append(nn.ReLU())
        for _ in range(self.num_layers_text_fc):
            neurons_per_layer_size_of_text_fc.append(nn.Linear(self.neurons_per_layer_size_of_text_fc, self.neurons_per_layer_size_of_text_fc))
            neurons_per_layer_size_of_text_fc.append(nn.ReLU())        
        neurons_per_layer_size_of_text_fc.append(nn.Linear(self.neurons_per_layer_size_of_text_fc, self.text_encoder_dim_output))
        
        # Montando o bloco de layers para processar os dados em One-Hot-Enconding
        self.text_fc = nn.Sequential(*neurons_per_layer_size_of_text_fc)
            
        # Projeções para espaço comum
        self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)
        self.text_projector = nn.Linear(self.text_encoder_dim_output, self.common_dim)

        # === Camada de Fusão Final ===
        self.fc_fusion = self.fc_mlp_module(n=1 if self.attention_mecanism in ["no-metadata", "att-intramodal+residual+cross-attention-metadados+metablock"] else self.n)
        # -------------------------
        # 3) Atenções Intra e Inter
        # -------------------------

        self.image_self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=False
        )

        self.text_self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=False
        )
        self.image_cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=False
        )

        self.text_cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim, 
            num_heads=self.num_heads,
            batch_first=False
        )
        # -------------------------
        # 4) Gating Mechanisms
        # -------------------------
        self.img_gate = nn.Linear(self.common_dim, self.common_dim)
        self.txt_gate = nn.Linear(self.common_dim, self.common_dim)
        # -------------------------
        # 6) Residual Blocks
        # -------------------------
        self.image_residual = GatedAlteredResidualBlock(dim=self.common_dim)
        self.text_residual = GatedAlteredResidualBlock(dim=self.common_dim)

        # Bloco do Metablock, caso queira usar
        self.meta_block = MetaBlock(V=self.common_dim if self.attention_mecanism in ["att-intramodal+residual+cross-attention-metadados+metablock"] else self.cnn_dim_output,
            U=self.common_dim if self.attention_mecanism in ["att-intramodal+residual+cross-attention-metadados+metablock"] else self.text_encoder_dim_output
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_no_mlp_to_visual_cls = nn.Sequential(
            nn.Linear(self.cnn_dim_output, self.num_classes),
            nn.Softmax(dim=1)
        )

    def fc_mlp_module(self, n=1):
        # Fusion block
        neurons_per_layer_size_of_fc_module = []
        neurons_per_layer_size_of_fc_module.append(nn.Linear(self.common_dim*self.n, self.neurons_per_layer_size_of_fc_module))
        neurons_per_layer_size_of_fc_module.append(nn.BatchNorm1d(self.neurons_per_layer_size_of_fc_module))
        neurons_per_layer_size_of_fc_module.append(nn.ReLU())
        neurons_per_layer_size_of_fc_module.append(nn.Dropout(0.3))

        for _ in range(self.num_layers_fc_module):
            neurons_per_layer_size_of_fc_module.append(nn.Linear(self.neurons_per_layer_size_of_fc_module, self.neurons_per_layer_size_of_fc_module))
            neurons_per_layer_size_of_fc_module.append(nn.ReLU())        
        
        # Último layer com a quantidade de classes
        neurons_per_layer_size_of_fc_module.append(nn.Linear(self.neurons_per_layer_size_of_fc_module, self.num_classes))
        neurons_per_layer_size_of_fc_module.append(nn.Softmax(dim=1))
        
        # Montando o bloco de layers para processar as features fundidas
        return nn.Sequential(*neurons_per_layer_size_of_fc_module)
            

        # return nn.Sequential(
        #     nn.Linear(self.common_dim*self.n, self.common_dim),
        #     nn.BatchNorm1d(self.common_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(self.common_dim, self.common_dim // 2),
        #     nn.BatchNorm1d(self.common_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(self.common_dim // 2, self.num_classes),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, image, text_metadata):
        # === Image Encoder ===
        image = image.to(self.device)
        # === [A] Image Feature Extraction ===
        # CNN -> (batch, cnn_dim_output)
        image_features = self.image_encoder(image) #.to(self.device)
        # Dá forma (batch, 1, cnn_dim_output)
        # Suaviza dimensões como (B, C, 1, 1) para (B, C)
        image_features = image_features.squeeze(-1).squeeze(-1)

        # Agora adiciona dimensão temporal (seq_len = 1)
        image_features_before = image_features.unsqueeze(1)  # (B, 1, C)

        #print(f"Image shape before: {image_features_before.shape}\n")
        # Projeção p/ espaço comum
        b_i, s_i, d_i = image_features_before.shape
        image_features_before = image_features_before.view(b_i*s_i, d_i)
        projected_image_features = self.image_projector(image_features_before)
        image_features = projected_image_features.view(b_i, s_i, -1)
        # -> (seq_len_img, batch, common_dim)
        image_features = image_features.permute(1, 0, 2)

        # === Text Encoder ===
        if self.text_model_name in ["one-hot-encoder", "tab-transformer"]:
            text_features = self.text_fc(text_metadata)  # (batch, 512)
            text_features = text_features.squeeze(-1).squeeze(-1)

            # Agora adiciona dimensão temporal (seq_len = 1)
            text_features = text_features.unsqueeze(1)  # (B, 1, C)

            # Projeção para espaço comum
            b_tt, s_tt, d_tt = text_features.shape
            before_project_text_features = text_features.view(b_tt*s_tt, d_tt)
            projected_text_features = self.text_projector(before_project_text_features)
            text_features = projected_text_features.view(b_tt, s_tt, -1)
            text_features = text_features.permute(1, 0, 2)
        else:
            raise ValueError("Encoder de texto não implementado!")

        # === [C] Self-Attention Intra-Modality ===
        image_features_att, _ = self.image_self_attention(
            image_features, image_features, image_features
        )
        text_features_att, _ = self.text_self_attention(
            text_features, text_features, text_features
        )

        # === [D] Cross-Attention Inter-Modality ===
        # "Imagem assiste ao texto"
        image_cross_att, _ = self.image_cross_attention(
            query=image_features_att,
            key=text_features_att,
            value=text_features_att
        )
        # "Texto assiste à imagem"
        text_cross_att, _ = self.text_cross_attention(
            query=text_features_att,
            key=image_features_att,
            value=image_features_att
        )

        # === [E] Pooling das atenções finais 
        image_cross_att = image_cross_att.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
        text_cross_att = text_cross_att.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

        image_pooled = image_cross_att.mean(dim=1)  # (batch, common_dim)
        text_pooled = text_cross_att.mean(dim=1)    # (batch, common_dim)
        
        if self.attention_mecanism=="metablock":
            meta_block_features = self.meta_block(image_features_before, before_project_text_features)  # [B, num_channels, H', W']
            # Pooling global e classificação
            pooled_features = self.avg_pool(meta_block_features)  # [B, num_channels, 1, 1]
            pooled_features = pooled_features.view(pooled_features.size(0), -1)  # [B, num_channels]
            return self.fc_no_mlp_to_visual_cls(pooled_features)
    
        elif self.attention_mecanism=="no-metadata":
            combined_features = projected_image_features
            output = self.fc_fusion(combined_features)  # (batch, num_classes)
            return output
        elif self.attention_mecanism=="no-metadata-without-mlp":
            output = self.fc_no_mlp_to_visual_cls(image_features_before)
            return output
        elif self.attention_mecanism == "weighted":
            # # === [F] Gating: quanto usar de 'peso' para cada modal
            # projected_image_features = projected_image_features.squeeze(0)
            # projected_text_features = projected_text_features.squeeze(0)
            alpha_img = torch.sigmoid(self.img_gate(projected_image_features))  # (batch, common_dim)
            alpha_txt = torch.sigmoid(self.txt_gate(projected_text_features))   # (batch, common_dim)

            # Multiplicamos as features pela máscara gerada
            image_pooled_gated = alpha_img * projected_image_features
            text_pooled_gated = alpha_txt * projected_text_features
            combined_features = torch.cat([image_pooled_gated, text_pooled_gated], dim=1)
            output = self.fc_fusion(combined_features)  # (batch, num_classes)
            return output
        elif self.attention_mecanism == "concatenation":
            # # Apenas concatena as features projetadas
            combined_features = torch.cat((projected_image_features.squeeze(0), projected_text_features.squeeze(0)), dim=-1)
            output = self.fc_fusion(combined_features)
            return output
        elif self.attention_mecanism == "gfcam":
            # # === [F] Gating: quanto usar de cada modal?
            #  Após o uso de cross-attention, as features são multiplicadas por cada fator individual de cada modalidade
            alpha_img = torch.sigmoid(self.img_gate(image_pooled))  # (batch, common_dim)
            alpha_txt = torch.sigmoid(self.txt_gate(text_pooled))   # (batch, common_dim)
            # Multiplicamos as features pela máscara gerada
            image_pooled_gated = alpha_img * image_pooled
            text_pooled_gated = alpha_txt * text_pooled
            # === [G] Fusão e classificação
            combined_features = torch.cat([image_pooled_gated, text_pooled_gated], dim=1)
            output = self.fc_fusion(combined_features)  # (batch, num_classes)
            return output
        elif self.attention_mecanism == "crossattention":
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)
            output = self.fc_fusion(combined_features)  # (batch, num_classes)
            return output
        elif self.attention_mecanism == "cross-weights-after-crossattention":
            #  Após o uso de cross-attention, as features são multiplicadas por cada fator individual de cada modalidade
            alpha_img = torch.sigmoid(self.img_gate(image_pooled))  # (batch, common_dim)
            alpha_txt = torch.sigmoid(self.txt_gate(text_pooled))   # (batch, common_dim)
            # Multiplicamos as features pela máscara gerada
            image_pooled_gated = alpha_txt * image_pooled
            text_pooled_gated = alpha_img * text_pooled
            # === [G] Fusão e classificação
            combined_features = torch.cat([image_pooled_gated, text_pooled_gated], dim=1)
            output = self.fc_fusion(combined_features)  # (batch, num_classes)
            return output
        # Com blocos residuais e gated
        elif self.attention_mecanism=="att-intramodal+residual":
            # === Self-Attention Intra-Modality ===
            image_features_att, _ = self.image_self_attention(
                image_features, image_features, image_features
            )

            text_features_att, _ = self.text_self_attention(
                text_features, text_features, text_features
            )
            # Bloco residual
            image_features_residual_before_cross_attention = self.image_residual(image_features, image_features_att, image_features_att)
            text_features_residual_before_cross_attention = self.text_residual(text_features, text_features_att, text_features_att)
            
            # === Pooling das features finais 
            image_features_residual_before_cross_attention = image_features_residual_before_cross_attention.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
            text_features_residual_before_cross_attention = text_features_residual_before_cross_attention.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

            image_pooled = image_features_residual_before_cross_attention.mean(dim=1)  # (batch, common_dim)
            text_pooled = text_features_residual_before_cross_attention.mean(dim=1)    # (batch, common_dim)
            # === Fusão das features
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)
            output = self.fc_fusion(combined_features)  # (batch, num_classes)
            return output
        elif self.attention_mecanism=="att-intramodal+residual+cross-attention-metadados":
            # === Self-Attention Intra-Modality ===
            image_features_att, _ = self.image_self_attention(
                image_features, image_features, image_features
            )

            text_features_att, _ = self.text_self_attention(
                text_features, text_features, text_features
            )
            # Bloco residual
            image_features_residual_before_cross_attention = self.image_residual(image_features, image_features_att, image_features_att)
            text_features_residual_before_cross_attention = self.text_residual(text_features, text_features_att, text_features_att)

            # === Cross-Attention Inter-Modality ===
            image_cross_att, _ = self.image_cross_attention(
                query=image_features_residual_before_cross_attention,
                key=text_features_residual_before_cross_attention,
                value=text_features_residual_before_cross_attention
            )
            # "Texto assiste à imagem"
            text_cross_att, _ = self.text_cross_attention(
                query=text_features_residual_before_cross_attention,
                key=image_features_residual_before_cross_attention,
                value=image_features_residual_before_cross_attention
            )
            # === Pooling das features finais 
            image_cross_att = image_cross_att.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
            text_cross_att = text_cross_att.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

            image_pooled = image_cross_att.mean(dim=1)  # (batch, common_dim)
            text_pooled = text_cross_att.mean(dim=1)    # (batch, common_dim)
            # === Fusão das features
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)
            output = self.fc_fusion(combined_features)  # (batch, num_classes)
            return output
        elif self.attention_mecanism=="att-intramodal+residual+cross-attention-metadados+metablock":
            # === Self-Attention Intra-Modality ===
            image_features_att, _ = self.image_self_attention(
                image_features, image_features, image_features
            )

            text_features_att, _ = self.text_self_attention(
                text_features, text_features, text_features
            )
            # Bloco residual
            image_features_residual_before_cross_attention = self.image_residual(image_features, image_features_att, image_features_att)
            text_features_residual_before_cross_attention = self.text_residual(text_features, text_features_att, text_features_att)

            # === Cross-Attention Inter-Modality ===
            image_cross_att, _ = self.image_cross_attention(
                query=image_features_residual_before_cross_attention,
                key=text_features_residual_before_cross_attention,
                value=text_features_residual_before_cross_attention
            )
            # "Texto assiste à imagem"
            text_cross_att, _ = self.text_cross_attention(
                query=text_features_residual_before_cross_attention,
                key=image_features_residual_before_cross_attention,
                value=image_features_residual_before_cross_attention
            )
            # === Pooling das features finais 
            image_cross_att = image_cross_att.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
            text_cross_att = text_cross_att.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

            image_pooled = image_cross_att.mean(dim=1)  # (batch, common_dim)
            text_pooled = text_cross_att.mean(dim=1)    # (batch, common_dim)
            # === Fusão das features
            meta_block_features = self.meta_block(image_pooled, text_pooled)  # [B, num_channels, H', W']
            # Pooling global e classificação
            pooled_features = self.avg_pool(meta_block_features)  # [B, num_channels, 1, 1]
            pooled_metablock_features = pooled_features.view(pooled_features.size(0), -1)  # [B, num_channels]
            output = self.fc_fusion(pooled_metablock_features)  # (batch, num_classes)
            return output
        elif self.attention_mecanism=="att-intramodal+residual+cross-attention-metadados+att-intramodal+residual":
            # === Self-Attention Intra-Modality ===
            image_features_att, _ = self.image_self_attention(
                image_features, image_features, image_features
            )

            text_features_att, _ = self.text_self_attention(
                text_features, text_features, text_features
            )
            # Bloco residual
            image_features_residual_before_cross_attention = self.image_residual(image_features, image_features_att, image_features_att)
            text_features_residual_before_cross_attention = self.text_residual(text_features, text_features_att, text_features_att)

            # === Cross-Attention Inter-Modality ===
            image_cross_att, _ = self.image_cross_attention(
                query=image_features_residual_before_cross_attention,
                key=text_features_residual_before_cross_attention,
                value=text_features_residual_before_cross_attention
            )
            # "Texto assiste à imagem"
            text_cross_att, _ = self.text_cross_attention(
                query=image_features_residual_before_cross_attention,
                key=image_features_residual_before_cross_attention,
                value=text_features_residual_before_cross_attention
            )

            # === Self-Attention Intra-Modality after cross-attention ===
            image_features_att_after_cross_att, _ = self.image_self_attention(
                image_cross_att, image_cross_att, image_cross_att
            )

            text_features_att_after_cross_att, _ = self.text_self_attention(
                text_cross_att, text_cross_att, text_cross_att
            )
            # Bloco residual
            image_features_residual_after_cross_attention = self.image_residual(image_features, image_features_att_after_cross_att, image_features_att_after_cross_att)
            text_features_residual_after_cross_attention = self.text_residual(text_features, text_features_att_after_cross_att, text_features_att_after_cross_att)
            
            # === Pooling das features finais 
            image_features_residual_after_cross_attention = image_features_residual_after_cross_attention.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
            text_features_residual_after_cross_attention = text_features_residual_after_cross_attention.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

            image_pooled = image_features_residual_after_cross_attention.mean(dim=1)  # (batch, common_dim)
            text_pooled = text_features_residual_after_cross_attention.mean(dim=1)    # (batch, common_dim)
            # === Fusão das features
            combined_features = torch.cat([image_pooled, text_pooled], dim=1)
            output = self.fc_fusion(combined_features)  # (batch, num_classes)
            return output