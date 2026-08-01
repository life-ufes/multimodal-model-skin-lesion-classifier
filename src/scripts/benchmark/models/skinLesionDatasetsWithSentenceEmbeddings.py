"""Dataset que entrega *sentence embeddings* pré-computados como metadados.

Feito para o `multimodalintraintermodalemb.MultimodalModel`, cujo ramo textual
aceita um tensor float (B, D) quando o encoder é de embeddings pré-computados.

Por que pré-computar em vez de rodar o encoder a cada batch:

- Com `UNFREEZE_WEIGHTS=frozen_weights` o encoder textual não é treinado, então o
  embedding de cada sentença é constante — recalcular a cada época é desperdício.
- Uma passada única sobre o CSV é ordens de magnitude mais rápida que N épocas.
- O encoder textual não fica residente na GPU durante o treino, o que libera VRAM
  (relevante em placas de 4 GB).

Os vetores vão para um cache em disco chaveado por (CSV, mtime, tamanho, embedder,
coluna de texto), de modo que trocar de embedder ou de CSV invalida o cache
automaticamente.
"""

import hashlib
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

# Prefixo que o multimodalintraintermodalemb reconhece como "embedding pronto"
# (ver MultimodalModel.SENTENCE_EMBEDDING_ENCODERS).
SENTENCE_EMBEDDING_PREFIX = "sentence-embedding:"

# Aliases -> id no HuggingFace. A dimensão NÃO é fixada aqui: é lida do próprio
# modelo (`get_sentence_embedding_dimension`), evitando que registro e realidade
# saiam de sincronia.
SENTENCE_EMBEDDERS = {
    "pubmedbert-base-embeddings": "neuml/pubmedbert-base-embeddings",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
}


def resolve_embedder_name(text_model_encoder: str) -> str:
    """Extrai o alias do embedder a partir do nome do encoder de texto.

    Aceita tanto 'sentence-embedding:<alias>' quanto o alias nu, e tolera os
    sufixos históricos do PubMedBERT ('pubmedbert-base-embeddings-500K'), que
    designavam o mesmo modelo.
    """
    name = str(text_model_encoder)
    if name.startswith(SENTENCE_EMBEDDING_PREFIX):
        name = name[len(SENTENCE_EMBEDDING_PREFIX):]

    if name in SENTENCE_EMBEDDERS:
        return name

    if name.startswith("pubmedbert-base-embeddings"):
        return "pubmedbert-base-embeddings"

    raise ValueError(
        f"Sentence-embedder '{text_model_encoder}' não registrado. "
        f"Disponíveis: {sorted(SENTENCE_EMBEDDERS)}"
    )


class SkinLesionDataset(Dataset):
    """Imagem + sentence embedding pré-computado + rótulo."""

    def __init__(
        self,
        metadata_file,
        img_dir,
        bert_model_name,
        drop_nan=False,
        image_encoder="resnet-18",
        size=(224, 224),
        is_train=False,
        sentence_column="sentence",
        cache_dir=None,
        encode_batch_size=64,
        encode_device=None,
    ):
        self.metadata_file = metadata_file
        self.img_dir = img_dir
        self.is_to_drop_nan = drop_nan
        self.image_encoder = image_encoder
        self.size = size
        self.is_train = is_train
        self.sentence_column = sentence_column
        self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.targets = None

        self.embedder_alias = resolve_embedder_name(bert_model_name)
        self.embedder_id = SENTENCE_EMBEDDERS[self.embedder_alias]

        self.transform = self.load_transforms()
        self.metadata = self.load_metadata()
        self.labels = self.metadata['diagnostic'].astype('category').cat.codes.tolist()

        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(metadata_file)), "sentence_embeddings_cache"
        )
        self.embeddings = self._load_or_compute_embeddings(
            encode_batch_size, encode_device
        )
        self.embedding_dim = int(self.embeddings.shape[1])

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def _cache_key(self):
        stat = os.stat(self.metadata_file)
        payload = "|".join([
            os.path.abspath(self.metadata_file),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            self.embedder_id,
            self.sentence_column,
            str(len(self.metadata)),
        ])
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.embedder_alias}_{digest}.npy"

    def _load_or_compute_embeddings(self, batch_size, device):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, self._cache_key())

        if os.path.isfile(cache_path):
            emb = np.load(cache_path)
            if emb.shape[0] == len(self.metadata):
                print(f"Sentence embeddings do cache: {cache_path} {emb.shape}")
                return emb.astype(np.float32)
            print(f"Cache com tamanho incompatível ({emb.shape[0]} != "
                  f"{len(self.metadata)}), recalculando: {cache_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sentences = (
            self.metadata[self.sentence_column]
            .fillna("")
            .astype(str)
            .tolist()
        )
        print(f"Calculando {len(sentences)} sentence embeddings com "
              f"'{self.embedder_id}' em {device}...")

        model = SentenceTransformer(self.embedder_id, device=device)
        emb = model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Libera o encoder: durante o treino ele não é necessário, e manter os
        # pesos na GPU competiria por VRAM com o backbone de imagem.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        np.save(cache_path, emb)
        print(f"Sentence embeddings salvos em {cache_path} {emb.shape}")
        return emb

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_name = self.metadata.iloc[idx]['img_id']
        img_path = os.path.abspath(os.path.join(self.img_dir, image_name))

        try:
            with Image.open(img_path) as img:
                image = np.array(img.convert("RGB"))
        except Exception as e:
            print(f"[Erro] Não foi possível abrir imagem com PIL: {img_path} — {e}")
            raise FileNotFoundError(f"Imagem inválida: {img_path}")

        if self.transform:
            image = self.transform(image=image)['image']

        embedding = torch.from_numpy(self.embeddings[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_name, image, embedding, label

    def load_transforms(self):
        if self.is_train:
            drop_prob = np.random.uniform(0.0, 0.05)
            return A.Compose([
                A.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)}, p=0.25),
                A.Resize(self.size[0], self.size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(rotate=(-120, 120), mode=cv2.BORDER_REFLECT, p=0.25),
                A.GaussianBlur(sigma_limit=(0, 3.0), p=0.25),
                A.OneOf([
                    A.PixelDropout(dropout_prob=drop_prob, p=1),
                    A.CoarseDropout(
                        num_holes_range=(int(0.00125 * self.size[0] * self.size[1]),
                                         int(0.00125 * self.size[0] * self.size[1])),
                        hole_height_range=(4, 4),
                        hole_width_range=(4, 4),
                        p=1),
                ], p=0.1),
                A.OneOf([
                    A.OneOrOther(
                        first=A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, p=1),
                        second=A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=False, p=1),
                        p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=0, p=1),
                ], p=0.25),
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2(),
            ])
        return A.Compose([
            A.Resize(self.size[0], self.size[1]),
            A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
            ToTensorV2()
        ])

    def load_metadata(self):
        metadata = pd.read_csv(self.metadata_file).fillna("EMPTY").replace(
            " ", "EMPTY").replace("  ", "EMPTY").replace(
            "NÃO  ENCONTRADO", "EMPTY").replace("BRASIL", "BRAZIL")

        if self.sentence_column not in metadata.columns:
            raise ValueError(
                f"Coluna '{self.sentence_column}' ausente em {self.metadata_file}. "
                f"Colunas: {list(metadata.columns)}"
            )

        # CORREÇÃO (Parte 4): os rótulos vêm de `.astype('category').cat.codes`,
        # que ordena as categorias alfabeticamente. `unique()` devolve na ordem
        # de aparição no CSV, desalinhando nomes de classe nos relatórios e na
        # matriz de confusão.
        self.targets = list(metadata['diagnostic'].astype('category').cat.categories)
        if self.is_to_drop_nan is True:
            metadata = metadata.dropna().reset_index(drop=True)
        return metadata