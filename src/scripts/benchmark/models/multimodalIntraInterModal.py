import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gatedResidualBlock import GatedAlteredResidualBlock, StackedGatedResidualBlock
from loadImageModelClassifier import loadModels
from metablock import MetaBlock
from metanet import MetaNet
from multimodalMDNet import MDNet

class MultimodalModel(nn.Module):
    def __init__(
        self,
        num_classes,
        num_heads,
        device,
        cnn_model_name,
        text_model_name,
        batch_size=32,
        common_dim=512,
        text_encoder_dim_output=512,
        vocab_size=91,
        unfreeze_weights="frozen",
        attention_mecanism="concatenation",
        n=2
    ):
        super().__init__()

        self.device = device
        self.common_dim = common_dim
        self.num_heads = num_heads
        self.attention_mecanism = attention_mecanism
        self.n = n
        self.vocab_size = vocab_size 
        self.num_classes = num_classes
        self.cnn_model_name = cnn_model_name
        self.text_model_name = text_model_name
        self.unfreeze_weights = unfreeze_weights
        # =====================================================
        # Text Encoder
        # =====================================================
        self.text_encoder_dim_output = text_encoder_dim_output

        # =====================================================
        # Image Encoder
        # =====================================================
        self.image_encoder, self.cnn_dim_output = loadModels.loadModelImageEncoder(
            cnn_model_name=self.cnn_model_name,
            backbone_train_mode=self.unfreeze_weights
        )

        self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)

        if text_model_name == "one-hot-encoder":
            self.text_fc = nn.Sequential(
                nn.Linear(self.vocab_size, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, self.text_encoder_dim_output)
            )
            self.text_encoder = None
        else:
            self.text_encoder, self.text_encoder_dim_output, _ = loadModels.loadTextModelEncoder(
                text_model_encoder=self.text_model_name,
                train_mode=self.unfreeze_weights
            )
            self.text_fc = None

        self.text_projector = nn.Linear(self.text_encoder_dim_output, self.common_dim)

        # =====================================================
        # Attention blocks
        # =====================================================
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

        # =====================================================
        # Gating
        # =====================================================
        self.img_gate = nn.Linear(self.common_dim, self.common_dim)
        self.txt_gate = nn.Linear(self.common_dim, self.common_dim)

        # =====================================================
        # MetaBlock (vetorial)
        # =====================================================
        # Bloco do Metablock, caso queira usar 
        self.meta_block = MetaBlock(
            V_dim=self.common_dim if self.attention_mecanism in ["att-intramodal+residual+cross-attention-metadados+metablock"] else self.cnn_dim_output,
            U_dim=self.common_dim if self.attention_mecanism in ["att-intramodal+residual+cross-attention-metadados+metablock", "metablock-se"] else self.text_encoder_dim_output 
        )
        # Residual Blocks
        # -------------------------
        self.image_residual = GatedAlteredResidualBlock(dim=self.common_dim)
        self.text_residual = GatedAlteredResidualBlock(dim=self.common_dim)

        # =====================================================
        # Fusion MLP
        # =====================================================
        self.fc_fusion = self.fc_mlp_module(
            n=1 if attention_mecanism == "no-metadata" else self.n
        )

        # Usado no Metablock
        self.fc_visual_only = nn.Linear(self.cnn_dim_output, self.num_classes)

        self.fc_mlp_module_after_metablock_fusion_module = self.fc_mlp_module_after_metablock()

    def fc_mlp_module(self, n=1):
        fc_fusion = nn.Sequential(
            nn.Linear(self.common_dim * n, self.common_dim),
            nn.LayerNorm(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.LayerNorm(self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.common_dim // 2, self.num_classes)
        )
        return fc_fusion
    
    def fc_mlp_module_after_metablock(self):
        fc_fusion = nn.Sequential(
            nn.Linear(self.cnn_dim_output, self.common_dim),
            nn.LayerNorm(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.LayerNorm(self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim // 2, self.num_classes)
        )
        return fc_fusion

    def forward(self, image, text_metadata):
        # =====================================================
        # Image encoding → (B, D)
        # =====================================================
        image = image.to(self.device)
        img_feat = self.image_encoder(image)

        if img_feat.dim() == 4:
            img_feat = img_feat.mean(dim=(-2, -1))

        proj_img_feat = self.image_projector(img_feat)

        # =====================================================
        # Text encoding → (B, D)
        # =====================================================
        if self.text_model_name == "one-hot-encoder":
            txt_feat = self.text_fc(text_metadata.to(self.device))
        else:
            input_ids = text_metadata["input_ids"].squeeze(1).to(self.device)
            attention_mask = text_metadata["attention_mask"].squeeze(1).to(self.device)
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            txt_feat = outputs.last_hidden_state[:, 0, :]

        proj_txt_feat = self.text_projector(txt_feat)

        # =====================================================
        # Attention (seq_len = 1)
        # =====================================================
        img_seq = proj_img_feat.unsqueeze(0)
        txt_seq = proj_txt_feat.unsqueeze(0)

        img_att, _ = self.image_self_attention(img_seq, img_seq, img_seq)
        txt_att, _ = self.text_self_attention(txt_seq, txt_seq, txt_seq)

        img_cross, _ = self.image_cross_attention(img_att, txt_att, txt_att)
        txt_cross, _ = self.text_cross_attention(txt_att, img_att, img_att)

        img_pooled = img_cross.squeeze(0)
        txt_pooled = txt_cross.squeeze(0)

        # =====================================================
        # Fusion strategies
        # =====================================================
        if self.attention_mecanism == "no-metadata":
            return self.fc_fusion(proj_img_feat)

        elif self.attention_mecanism == "no-metadata-without-mlp":
            return self.fc_visual_only(img_seq)

        elif self.attention_mecanism == "concatenation":
            fused = torch.cat([proj_img_feat, proj_txt_feat], dim=1)
            return self.fc_fusion(fused)

        elif self.attention_mecanism == "crossattention":
            fused = torch.cat([img_pooled, txt_pooled], dim=1)
            return self.fc_fusion(fused)

        elif self.attention_mecanism == "weighted":
            alpha_img = torch.sigmoid(self.img_gate(proj_img_feat))
            alpha_txt = torch.sigmoid(self.txt_gate(proj_txt_feat))
            fused = torch.cat([alpha_img * proj_img_feat, alpha_txt * proj_txt_feat], dim=1)
            return self.fc_fusion(fused)

        elif self.attention_mecanism == "gfcam":
            alpha_img = torch.sigmoid(self.img_gate(img_pooled))
            alpha_txt = torch.sigmoid(self.txt_gate(txt_pooled))
            fused = torch.cat([alpha_img * img_pooled, alpha_txt * txt_pooled], dim=1)
            return self.fc_fusion(fused)

        elif self.attention_mecanism == "cross-weights-after-crossattention":
            alpha_img = torch.sigmoid(self.img_gate(img_pooled))
            alpha_txt = torch.sigmoid(self.txt_gate(txt_pooled))
            fused = torch.cat([alpha_txt * img_pooled, alpha_img * txt_pooled], dim=1)
            return self.fc_fusion(fused)
        
        elif self.attention_mecanism == "metablock":
            meta_features = self.meta_block(
                img_feat,      
                txt_feat
            )
            return self.fc_mlp_module_after_metablock_fusion_module(meta_features)
        
        # =====================================================
        # Residual-based multimodal fusion
        # =====================================================

        # =====================================================
        # Residual-based multimodal fusion (corrigidos)
        # =====================================================

        elif self.attention_mecanism == "only-with-att-intramodal+residual":
            # Residual direto (sem self-att explícito)
            # No nosso forward atual, a forma padrão é (seq_len=1, B, D) => img_seq/txt_seq
            img_res = self.image_residual(img_seq, txt_seq, txt_seq)  # (1, B, D)
            txt_res = self.text_residual(txt_seq, img_seq, img_seq)   # (1, B, D)

            img_res = img_res.squeeze(0)  # (B, D)
            txt_res = txt_res.squeeze(0)  # (B, D)

            fused = torch.cat([img_res, txt_res], dim=1)  # (B, 2D)
            return self.fc_fusion(fused)

        elif self.attention_mecanism == "att-intramodal+residual":
            # Self-att já calculado acima: img_att, txt_att
            # Residual usa (base, att, att)
            img_res = self.image_residual(img_seq, img_att, img_att)  # (1, B, D)
            txt_res = self.text_residual(txt_seq, txt_att, txt_att)   # (1, B, D)

            img_res = img_res.squeeze(0)  # (B, D)
            txt_res = txt_res.squeeze(0)  # (B, D)

            fused = torch.cat([img_res, txt_res], dim=1)  # (B, 2D)
            return self.fc_fusion(fused)

        elif self.attention_mecanism == "att-intramodal+residual+cross-attention-metadados":
            # Self-att já calculado: img_att, txt_att
            # Residual antes do cross
            img_res = self.image_residual(img_seq, img_att, img_att)  # (1, B, D)
            txt_res = self.text_residual(txt_seq, txt_att, txt_att)   # (1, B, D)

            # Cross-attention entre os residuais
            img_cross2, _ = self.image_cross_attention(
                query=img_res, key=txt_res, value=txt_res
            )  # (1, B, D)

            txt_cross2, _ = self.text_cross_attention(
                query=txt_res, key=img_res, value=img_res
            )  # (1, B, D)

            img_pooled2 = img_cross2.squeeze(0)  # (B, D)
            txt_pooled2 = txt_cross2.squeeze(0)  # (B, D)

            fused = torch.cat([img_pooled2, txt_pooled2], dim=1)  # (B, 2D)
            return self.fc_fusion(fused)

        elif self.attention_mecanism == "att-intramodal+residual+cross-attention-metadados+metablock":
            # Self-att já calculado: img_att, txt_att
            # Residual antes do cross
            img_res = self.image_residual(img_seq, img_att, img_att)  # (1, B, D)
            txt_res = self.text_residual(txt_seq, txt_att, txt_att)   # (1, B, D)

            # Cross-attention
            img_cross2, _ = self.image_cross_attention(
                query=img_res, key=txt_res, value=txt_res
            )  # (1, B, D)

            txt_cross2, _ = self.text_cross_attention(
                query=txt_res, key=img_res, value=img_res
            )  # (1, B, D)

            img_pooled2 = img_cross2.squeeze(0)  # (B, D)
            txt_pooled2 = txt_cross2.squeeze(0)  # (B, D)

            # MetaBlock vetorial (B, D) + (B, D) -> (B, D)
            fused_meta = self.meta_block(img_pooled2, txt_pooled2)  # (B, D)

            # Classificador no espaço comum
            return self.fc_visual_only(fused_meta)

        elif self.attention_mecanism == "att-intramodal+residual+cross-attention-metadados+att-intramodal+residual":
            # Self-att inicial: img_att, txt_att
            # Residual antes do cross
            img_res1 = self.image_residual(img_seq, img_att, img_att)  # (1, B, D)
            txt_res1 = self.text_residual(txt_seq, txt_att, txt_att)   # (1, B, D)

            # Cross-attention
            img_cross2, _ = self.image_cross_attention(
                query=img_res1, key=txt_res1, value=txt_res1
            )  # (1, B, D)

            txt_cross2, _ = self.text_cross_attention(
                query=txt_res1, key=img_res1, value=img_res1
            )  # (1, B, D)

            # Self-att depois do cross
            img_att2, _ = self.image_self_attention(img_cross2, img_cross2, img_cross2)
            txt_att2, _ = self.text_self_attention(txt_cross2, txt_cross2, txt_cross2)

            # Residual final (base = cross2)
            img_res2 = self.image_residual(img_cross2, img_att2, img_att2).squeeze(0)  # (B, D)
            txt_res2 = self.text_residual(txt_cross2, txt_att2, txt_att2).squeeze(0)   # (B, D)

            fused = torch.cat([img_res2, txt_res2], dim=1)  # (B, 2D)
            return self.fc_fusion(fused)
        else:
            raise ValueError(
                f"Attention mechanism '{self.attention_mecanism}' not implemented."
            )
