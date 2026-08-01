"""Modelo multimodal (imagem + texto) com catálogo de mecanismos de fusão.

O ramo textual aceita duas formas de entrada:

- **embedding pré-computado** (`loadTextModelEncoder` devolve `None`): o dataset
  entrega um tensor float (B, D) já embutido — ver
  `skinLesionDatasetsWithSentenceEmbeddings`.
- **encoder HuggingFace residente**: o dataset entrega o dicionário com
  `input_ids`/`attention_mask` e o encoder roda a cada batch.

Os mecanismos de fusão têm os mesmos nomes de `multimodalIntraInterModal`, para
que os resultados dos dois scripts de benchmark sejam comparáveis.

Diferença de implementação em relação àquele módulo: aqui os blocos (self-att,
cross-att, gates, MetaBlock, residuais) são construídos **apenas** quando o
mecanismo escolhido os usa. Instanciar todos sempre inflaria a contagem de
parâmetros e os FLOPs reportados de runs que não os exercitam — o que, num
benchmark que compara arquiteturas, corrompe justamente a métrica de custo.
"""

import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loadImageModelClassifier import loadModels
from gatedResidualBlock import GatedAlteredResidualBlock
from metablock import MetaBlock


# Número de cabeças fixado dentro de GatedAlteredResidualBlock; `common_dim`
# precisa ser divisível por ele para os blocos residuais funcionarem.
RESIDUAL_BLOCK_NUM_HEADS = 8

# ---------------------------------------------------------------------------
# Catálogo de mecanismos
# ---------------------------------------------------------------------------
# Para cada mecanismo, quais componentes o forward exige. Esta tabela é a única
# fonte de verdade: o __init__ constrói a partir dela e o forward assume que o
# que ele usa foi construído. Acrescentar um mecanismo sem registrá-lo aqui
# falha na validação, não com um AttributeError no meio do treino.
#
#   self_att    - self-attention por modalidade
#   cross_att   - cross-attention entre modalidades
#   gates       - gating sigmoide (img_gate/txt_gate)
#   residual    - GatedAlteredResidualBlock por modalidade
#   metablock   - MetaBlock sobre features brutas (V=cnn_dim, U=text_dim)
#   metablock_common - MetaBlock no espaço comum (V=U=common_dim)
#
# `head` define o classificador:
#   fusion      - MLP sobre [img; txt] (2 * common_dim)
#   fusion_n1   - MLP sobre um único vetor (common_dim)
#   visual_only - Linear direto sobre as features brutas da CNN
#   proj_head   - Linear de common_dim -> num_classes
#   after_metablock - MLP de cnn_dim_output -> num_classes
MECHANISM_SPECS = {
    "no-metadata": {"needs": set(), "head": "fusion_n1"},
    "no-metadata-without-mlp": {"needs": set(), "head": "visual_only"},
    "concatenation": {"needs": set(), "head": "fusion"},
    "crossattention": {"needs": {"self_att", "cross_att"}, "head": "fusion"},
    "weighted": {"needs": {"gates"}, "head": "fusion"},
    "gfcam": {"needs": {"self_att", "cross_att", "gates"}, "head": "fusion"},
    "cross-weights-after-crossattention": {
        "needs": {"self_att", "cross_att", "gates"}, "head": "fusion"},
    "metablock": {"needs": {"metablock"}, "head": "after_metablock"},
    "rg-att2fusefeatures": {"needs": {"residual"}, "head": "proj_head"},
    "rg-att": {"needs": {"residual"}, "head": "fusion"},
    "att-intramodal": {"needs": {"self_att"}, "head": "fusion"},
    "att-intramodal+residual": {"needs": {"self_att", "residual"}, "head": "fusion"},
    "cross-attention-only": {"needs": {"cross_att"}, "head": "fusion"},
    "residual+cross-attention-metadados": {
        "needs": {"residual", "cross_att"}, "head": "fusion"},
    "att-intramodal+residual+cross-attention-metadados": {
        "needs": {"self_att", "residual", "cross_att"}, "head": "fusion"},
    "rg-att-literal-text-description": {"needs": {"residual"}, "head": "fusion"},
    "rg-att-cross-modal": {
        "needs": {"self_att", "residual", "cross_att"}, "head": "fusion"},
    "att-intramodal+residual+cross-attention-metadados+rg-att2fusefeatures": {
        "needs": {"self_att", "residual", "cross_att"}, "head": "proj_head"},
    "att-intramodal+residual+cross-attention-metadados+metablock": {
        "needs": {"self_att", "residual", "cross_att", "metablock_common"},
        "head": "proj_head"},
    "att-intramodal+residual+cross-attention-metadados+att-intramodal+residual": {
        "needs": {"self_att", "residual", "cross_att"}, "head": "fusion"},
}

# Nomes próprios de arquiteturas que mapeiam para um mecanismo do catálogo.
# "rg-dermnet" é o modelo proposto (o RG-ATT completo do README:
# self-att -> RG-residual -> cross-att -> concat -> MLP) e já nomeia pastas de
# resultados (ex.: src/results/shap_plots/rg-dermnet). Usar o alias mantém esse
# nome nos logs/MLflow sem duplicar a topologia no catálogo.
MECHANISM_ALIASES = {
    "rg-dermnet": "att-intramodal+residual+cross-attention-metadados",
}

# Exportado para os scripts de treino validarem a lista de experimentos antes de
# instanciar qualquer coisa (evita descobrir o erro só no primeiro batch).
ATTENTION_MECANISMS = sorted([*MECHANISM_SPECS, *MECHANISM_ALIASES])

# Mecanismos que consomem as features *brutas* dos encoders e portanto dispensam
# a projeção para o espaço comum. Sem isso, um run de `metablock` carregaria
# ~1,2 M de parâmetros de projeção que nenhum caminho do forward toca.
NO_IMAGE_PROJECTION = {"no-metadata-without-mlp", "metablock"}
NO_TEXT_PROJECTION = {"no-metadata", "no-metadata-without-mlp", "metablock"}


class MultimodalModel(nn.Module):
    def __init__(self, num_classes, num_heads, device, cnn_model_name, text_model_name,
                 common_dim=512, vocab_size=85, unfreeze_weights=False,
                 attention_mecanism="combined", n=2):
        super(MultimodalModel, self).__init__()

        # Resolve alias -> mecanismo canônico. O nome pedido fica guardado para
        # inspeção, mas todo o roteamento interno usa o canônico.
        self.attention_mecanism_requested = attention_mecanism
        attention_mecanism = MECHANISM_ALIASES.get(attention_mecanism, attention_mecanism)

        if attention_mecanism not in MECHANISM_SPECS:
            raise ValueError(
                f"Attention mechanism '{attention_mecanism}' not implemented. "
                f"Disponíveis: {ATTENTION_MECANISMS}"
            )

        # Dimensões do modelo
        self.common_dim = common_dim
        self.text_encoder_dim_output = 512
        self.cnn_dim_output = 512
        self.device = device
        self.cnn_model_name = cnn_model_name
        self.text_model_name = text_model_name
        self.attention_mecanism = attention_mecanism
        self.num_heads = num_heads  # para MultiheadAttention
        self.n = n
        self.num_classes = num_classes
        self.unfreeze_weights_of_visual_feat_extractor = unfreeze_weights

        spec = MECHANISM_SPECS[attention_mecanism]
        self.needs = spec["needs"]
        self.head = spec["head"]

        # -------------------------
        # 1) Image Encoder
        # -------------------------
        self.image_encoder, self.cnn_dim_output = loadModels.loadModelImageEncoder(
            self.cnn_model_name,
            self.common_dim,
            backbone_train_mode=self.unfreeze_weights_of_visual_feat_extractor
        )

        # -------------------------
        # 2) Text Encoder
        # -------------------------
        # CORREÇÃO (Parte 1): `vocab_size` recebido no construtor carrega a
        # dimensão real do metadado textual (embedding_dim do dataset). O
        # unpacking de loadTextModelEncoder sobrescrevia esse valor pelo 64
        # hardcoded do ramo PubMedBERT, dimensionando o ramo textual errado.
        embedding_dim_from_dataset = vocab_size

        self.text_encoder, self.text_encoder_dim_output, vocab_size = loadModels.loadTextModelEncoder(
            text_model_encoder=self.text_model_name,
            train_mode=self.unfreeze_weights_of_visual_feat_extractor)

        # Encoder de embeddings pré-computados: loadTextModelEncoder devolve
        # None, não há tokenização nem módulo textual, e a dimensão de entrada
        # é a do vetor já calculado pelo dataset.
        self.use_precomputed_text_embedding = self.text_encoder is None
        if self.use_precomputed_text_embedding:
            self.text_encoder_dim_output = embedding_dim_from_dataset
            vocab_size = embedding_dim_from_dataset

        # -------------------------
        # 3) Projeções para o espaço comum
        # -------------------------
        if attention_mecanism not in NO_TEXT_PROJECTION:
            self.text_projector = nn.Sequential(
                nn.Linear(self.text_encoder_dim_output, common_dim),
                nn.ReLU(),
                nn.Linear(common_dim, common_dim)
            )
        if attention_mecanism not in NO_IMAGE_PROJECTION:
            self.image_projector = nn.Sequential(
                nn.Linear(self.cnn_dim_output, common_dim),
                nn.ReLU(),
                nn.Linear(common_dim, common_dim)
            )

        # -------------------------
        # 4) Blocos de fusão (só os que o mecanismo usa)
        # -------------------------
        self._validate_head_divisibility()

        if "self_att" in self.needs:
            self.image_self_attention = self._make_attention()
            self.text_self_attention = self._make_attention()

        if "cross_att" in self.needs:
            self.image_cross_attention = self._make_attention()
            self.text_cross_attention = self._make_attention()

        if "gates" in self.needs:
            self.img_gate = nn.Linear(self.common_dim, self.common_dim)
            self.txt_gate = nn.Linear(self.common_dim, self.common_dim)

        if "residual" in self.needs:
            self.image_residual = GatedAlteredResidualBlock(dim=self.common_dim)
            self.text_residual = GatedAlteredResidualBlock(dim=self.common_dim)

        # O MetaBlock "cru" opera antes das projeções (dimensões dos encoders);
        # o "common" opera depois do cross-attention, no espaço comum.
        if "metablock" in self.needs:
            self.meta_block = MetaBlock(
                V_dim=self.cnn_dim_output,
                U_dim=self.text_encoder_dim_output
            )
        elif "metablock_common" in self.needs:
            self.meta_block = MetaBlock(
                V_dim=self.common_dim,
                U_dim=self.common_dim
            )

        # -------------------------
        # 5) Cabeça de classificação
        # -------------------------
        if self.head == "fusion":
            self.fc_fusion = self.fc_mlp_module(n=self.n)
        elif self.head == "fusion_n1":
            self.fc_fusion = self.fc_mlp_module(n=1)
        elif self.head == "visual_only":
            self.fc_visual_only = nn.Linear(self.cnn_dim_output, self.num_classes)
        elif self.head == "proj_head":
            self.fc_fusion_proj_feat2output = nn.Linear(self.common_dim, self.num_classes)
        elif self.head == "after_metablock":
            self.fc_mlp_module_after_metablock_fusion_module = \
                self.fc_mlp_module_after_metablock()

    # ------------------------------------------------------------------
    # Construtores auxiliares
    # ------------------------------------------------------------------
    def _make_attention(self):
        return nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=self.num_heads,
            batch_first=False
        )

    def _validate_head_divisibility(self):
        """Falha cedo em combinações (common_dim, num_heads) impossíveis.

        Sem isso o erro só apareceria dentro de nn.MultiheadAttention, depois de
        já ter carregado o backbone e o dataset inteiro.
        """
        if {"self_att", "cross_att"} & self.needs:
            if self.common_dim % self.num_heads != 0:
                raise ValueError(
                    f"common_dim={self.common_dim} não é divisível por "
                    f"num_heads={self.num_heads} (exigência de nn.MultiheadAttention)."
                )
        if "residual" in self.needs:
            if self.common_dim % RESIDUAL_BLOCK_NUM_HEADS != 0:
                raise ValueError(
                    f"common_dim={self.common_dim} não é divisível por "
                    f"{RESIDUAL_BLOCK_NUM_HEADS}, o número de cabeças fixado em "
                    f"GatedAlteredResidualBlock."
                )

    def fc_mlp_module(self, n=1):
        fc_fusion = nn.Sequential(
            nn.Linear(self.common_dim * n, self.common_dim),
            nn.BatchNorm1d(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.BatchNorm1d(self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim // 2, self.num_classes)# ,
            # nn.Softmax(dim=1)
        )
        return fc_fusion

    def fc_mlp_module_after_metablock(self):
        fc_fusion = nn.Sequential(
            nn.Linear(self.cnn_dim_output, self.common_dim),
            nn.BatchNorm1d(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.BatchNorm1d(self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.common_dim // 2, self.num_classes)
        )
        return fc_fusion

    # ------------------------------------------------------------------
    # Extração de features
    # ------------------------------------------------------------------
    def encode_image(self, image):
        """Features visuais brutas (B, cnn_dim_output)."""
        img_feat = self.image_encoder(image.to(self.device))
        # Backbones que devolvem mapa de features (B, C, H, W) precisam de
        # pooling antes das camadas densas.
        if img_feat.dim() == 4:
            img_feat = img_feat.mean(dim=(-2, -1))
        return img_feat

    def encode_text(self, metadata):
        """Features textuais brutas (B, text_encoder_dim_output)."""
        # CORREÇÃO (Parte 2): o dataset de sentence embeddings entrega um tensor
        # float (B, D); apenas os encoders HuggingFace recebem o dicionário com
        # input_ids/attention_mask.
        if self.use_precomputed_text_embedding:
            return metadata.float().to(self.device)

        input_ids = metadata['input_ids'].squeeze(1)
        attention_mask = metadata['attention_mask'].squeeze(1)

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if "gpt2" in self.text_model_name.lower():
            # GPT-2 não tem [CLS]: usa-se o último token da sequência.
            text_features = text_outputs.last_hidden_state[:, -1, :]
        else:
            text_features = text_outputs.last_hidden_state[:, 0, :]  # token [CLS]
        return text_features.to(self.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, image, metadata):
        img_feat = self.encode_image(image)

        # Baselines só-imagem: o ramo textual nem é percorrido. Importa não só
        # pelo custo (com encoder HuggingFace residente seria uma passada
        # completa do BERT descartada) mas pela honestidade do baseline.
        if self.attention_mecanism == "no-metadata-without-mlp":
            return self.fc_visual_only(img_feat)
        if self.attention_mecanism == "no-metadata":
            return self.fc_fusion(self.image_projector(img_feat))

        txt_feat = self.encode_text(metadata)

        if self.attention_mecanism == "metablock":
            # MetaBlock vetorial sobre as features brutas: (B, V) modulado por (B, U)
            return self.fc_mlp_module_after_metablock_fusion_module(
                self.meta_block(img_feat, txt_feat)
            )

        proj_img_feat = self.image_projector(img_feat)
        proj_txt_feat = self.text_projector(txt_feat)

        if self.attention_mecanism == "concatenation":
            return self.fc_fusion(torch.cat([proj_img_feat, proj_txt_feat], dim=1))

        if self.attention_mecanism == "weighted":
            alpha_img = torch.sigmoid(self.img_gate(proj_img_feat))
            alpha_txt = torch.sigmoid(self.txt_gate(proj_txt_feat))
            fused = torch.cat([alpha_img * proj_img_feat, alpha_txt * proj_txt_feat], dim=1)
            return self.fc_fusion(fused)

        # Os demais mecanismos operam com sequência de comprimento 1:
        # nn.MultiheadAttention com batch_first=False espera (seq_len, B, D).
        img_seq = proj_img_feat.unsqueeze(0)
        txt_seq = proj_txt_feat.unsqueeze(0)

        if "self_att" in self.needs:
            img_att, _ = self.image_self_attention(img_seq, img_seq, img_seq)
            txt_att, _ = self.text_self_attention(txt_seq, txt_seq, txt_seq)

        if self.attention_mecanism == "att-intramodal":
            return self.fc_fusion(
                torch.cat([img_att.squeeze(0), txt_att.squeeze(0)], dim=1))

        if self.attention_mecanism in ("crossattention", "gfcam",
                                       "cross-weights-after-crossattention"):
            # Cross-attention sobre as saídas de self-attention.
            img_cross, _ = self.image_cross_attention(img_att, txt_att, txt_att)
            txt_cross, _ = self.text_cross_attention(txt_att, img_att, img_att)
            img_pooled = img_cross.squeeze(0)
            txt_pooled = txt_cross.squeeze(0)

            if self.attention_mecanism == "crossattention":
                return self.fc_fusion(torch.cat([img_pooled, txt_pooled], dim=1))

            alpha_img = torch.sigmoid(self.img_gate(img_pooled))
            alpha_txt = torch.sigmoid(self.txt_gate(txt_pooled))
            if self.attention_mecanism == "gfcam":
                fused = torch.cat([alpha_img * img_pooled, alpha_txt * txt_pooled], dim=1)
            else:
                # cross-weights: o gate de uma modalidade pondera a outra.
                fused = torch.cat([alpha_txt * img_pooled, alpha_img * txt_pooled], dim=1)
            return self.fc_fusion(fused)

        if self.attention_mecanism == "cross-attention-only":
            img_cross, _ = self.image_cross_attention(img_seq, txt_seq, txt_seq)
            txt_cross, _ = self.text_cross_attention(txt_seq, img_seq, img_seq)
            return self.fc_fusion(
                torch.cat([img_cross.squeeze(0), txt_cross.squeeze(0)], dim=1))

        # --------------------------------------------------------------
        # Fusões baseadas em RG-ATT (GatedAlteredResidualBlock)
        # --------------------------------------------------------------
        if self.attention_mecanism == "rg-att2fusefeatures":
            # Metadado como query, imagem como key/value; um único vetor sai.
            fused = self.image_residual(txt_seq, img_seq, img_seq).squeeze(0)
            return self.fc_fusion_proj_feat2output(fused)

        if self.attention_mecanism == "rg-att":
            # Residual direto (sem self-attention explícito).
            img_res = self.image_residual(img_seq, txt_seq, txt_seq).squeeze(0)
            txt_res = self.text_residual(txt_seq, img_seq, img_seq).squeeze(0)
            return self.fc_fusion(torch.cat([img_res, txt_res], dim=1))

        if self.attention_mecanism == "rg-att-literal-text-description":
            # Descrição literal do RG-ATT (Seção III-B2): metadado é query (Q),
            # visual fornece key (K) e value (V). Sem estágio intra-modal nem
            # cross-attention separados — RG-ATT *é* o mecanismo de fusão. O
            # ramo visual entra na concatenação como sua própria projeção, já
            # que o artigo define uma única direção.
            txt_rgatt = self.text_residual(txt_seq, img_seq, img_seq).squeeze(0)
            return self.fc_fusion(torch.cat([proj_img_feat, txt_rgatt], dim=1))

        if self.attention_mecanism == "att-intramodal+residual":
            img_res = self.image_residual(img_seq, img_att, img_att).squeeze(0)
            txt_res = self.text_residual(txt_seq, txt_att, txt_att).squeeze(0)
            return self.fc_fusion(torch.cat([img_res, txt_res], dim=1))

        if self.attention_mecanism == "residual+cross-attention-metadados":
            # Residual (auto-referente) antes do cross-attention.
            img_res = self.image_residual(img_seq, img_seq, img_seq)
            txt_res = self.text_residual(txt_seq, txt_seq, txt_seq)
            img_cross, _ = self.image_cross_attention(img_res, txt_res, txt_res)
            txt_cross, _ = self.text_cross_attention(txt_res, img_res, img_res)
            return self.fc_fusion(
                torch.cat([img_cross.squeeze(0), txt_cross.squeeze(0)], dim=1))

        # Os mecanismos restantes compartilham o mesmo tronco:
        # self-att -> residual -> cross-att. Só a cabeça (ou o pós-processamento)
        # muda, então o tronco é calculado uma vez.
        img_res = self.image_residual(img_seq, img_att, img_att)
        txt_res = self.text_residual(txt_seq, txt_att, txt_att)
        img_cross, _ = self.image_cross_attention(img_res, txt_res, txt_res)
        txt_cross, _ = self.text_cross_attention(txt_res, img_res, img_res)

        if self.attention_mecanism in ("att-intramodal+residual+cross-attention-metadados",
                                       "rg-att-cross-modal"):
            return self.fc_fusion(
                torch.cat([img_cross.squeeze(0), txt_cross.squeeze(0)], dim=1))

        if self.attention_mecanism == \
                "att-intramodal+residual+cross-attention-metadados+rg-att2fusefeatures":
            fused = self.image_residual(txt_cross, img_cross, img_cross).squeeze(0)
            return self.fc_fusion_proj_feat2output(fused)

        if self.attention_mecanism == \
                "att-intramodal+residual+cross-attention-metadados+metablock":
            # MetaBlock vetorial no espaço comum: (B, D) + (B, D) -> (B, D)
            fused_meta = self.meta_block(img_cross.squeeze(0), txt_cross.squeeze(0))
            return self.fc_fusion_proj_feat2output(fused_meta)

        if self.attention_mecanism == \
                "att-intramodal+residual+cross-attention-metadados+att-intramodal+residual":
            # Segundo estágio self-att + residual, com o cross como base.
            img_att2, _ = self.image_self_attention(img_cross, img_cross, img_cross)
            txt_att2, _ = self.text_self_attention(txt_cross, txt_cross, txt_cross)
            img_res2 = self.image_residual(img_cross, img_att2, img_att2).squeeze(0)
            txt_res2 = self.text_residual(txt_cross, txt_att2, txt_att2).squeeze(0)
            return self.fc_fusion(torch.cat([img_res2, txt_res2], dim=1))

        # MECHANISM_SPECS e o forward saíram de sincronia.
        raise ValueError(
            f"Attention mechanism '{self.attention_mecanism}' registrado em "
            f"MECHANISM_SPECS mas sem implementação no forward."
        )
