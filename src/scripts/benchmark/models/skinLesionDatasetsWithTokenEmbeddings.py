"""Dataset que entrega *token embeddings* pré-computados do PubMedBERT.

Diferença em relação a `skinLesionDatasetsWithSentenceEmbeddings`: aquele agrega
a sentença inteira num único vetor (B, D); este preserva a sequência de tokens,
entregando (T, D) por amostra mais uma máscara de padding.

Motivo: com um único token textual, `nn.MultiheadAttention` degenera — o softmax
sobre uma única key retorna 1.0 e a saída passa a independer da query. Todo o
estágio de cross-attention do RG-DermNet vira uma permutação linear entre as
modalidades. Com T tokens, a atenção imagem->texto volta a ponderar de fato
quais trechos da descrição clínica importam para cada lesão.

O cache em disco guarda (N, T, D) em float16 (metade do espaço; a precisão
extra é irrelevante depois do LayerNorm a jusante) e a máscara em bool. A chave
inclui `max_tokens`, então mudar esse parâmetro invalida o cache sozinho.
"""

import hashlib
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset

TOKEN_EMBEDDING_PREFIX = "token-embedding:"

TOKEN_EMBEDDERS = {
    "pubmedbert-base-embeddings": "neuml/pubmedbert-base-embeddings",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}


def resolve_embedder_name(text_model_encoder: str) -> str:
    name = str(text_model_encoder)
    if name.startswith(TOKEN_EMBEDDING_PREFIX):
        name = name[len(TOKEN_EMBEDDING_PREFIX):]
    if name in TOKEN_EMBEDDERS:
        return name
    if name.startswith("pubmedbert-base-embeddings"):
        return "pubmedbert-base-embeddings"
    raise ValueError(
        f"Token-embedder '{text_model_encoder}' não registrado. "
        f"Disponíveis: {sorted(TOKEN_EMBEDDERS)}"
    )


class SkinLesionDataset(Dataset):
    """Imagem + sequência de token embeddings + máscara + rótulo."""

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
        encode_batch_size=32,
        encode_device=None,
        max_tokens=48,
    ):
        self.metadata_file = metadata_file
        self.img_dir = img_dir
        self.is_to_drop_nan = drop_nan
        self.image_encoder = image_encoder
        self.size = size
        self.is_train = is_train
        self.sentence_column = sentence_column
        self.max_tokens = int(max_tokens)
        self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.targets = None

        self.embedder_alias = resolve_embedder_name(bert_model_name)
        self.embedder_id = TOKEN_EMBEDDERS[self.embedder_alias]

        self.transform = self.load_transforms()
        self.metadata = self.load_metadata()
        self.labels = self.metadata['diagnostic'].astype('category').cat.codes.tolist()

        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(metadata_file)), "token_embeddings_cache"
        )
        self.embeddings, self.masks = self._load_or_compute_embeddings(
            encode_batch_size, encode_device
        )
        self.embedding_dim = int(self.embeddings.shape[2])
        self.seq_len = int(self.embeddings.shape[1])

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
            f"T{self.max_tokens}",
        ])
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.embedder_alias}_tok_{digest}.npz"

    def _load_or_compute_embeddings(self, batch_size, device):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, self._cache_key())

        if os.path.isfile(cache_path):
            data = np.load(cache_path)
            emb, mask = data["emb"], data["mask"]
            if emb.shape[0] == len(self.metadata):
                print(f"Token embeddings do cache: {cache_path} {emb.shape}")
                return emb, mask
            print(f"Cache incompatível ({emb.shape[0]} != {len(self.metadata)}), "
                  f"recalculando: {cache_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sentences = (
            self.metadata[self.sentence_column].fillna("").astype(str).tolist()
        )
        print(f"Calculando token embeddings de {len(sentences)} sentenças com "
              f"'{self.embedder_id}' em {device} (max_tokens={self.max_tokens})...")

        model = SentenceTransformer(self.embedder_id, device=device)
        # output_value='token_embeddings' devolve uma lista de tensores
        # (seq_len_i, D) — comprimento variável, incluindo tokens especiais.
        per_sentence = model.encode(
            sentences,
            batch_size=batch_size,
            output_value="token_embeddings",
            convert_to_numpy=False,
            show_progress_bar=True,
        )

        dim = int(per_sentence[0].shape[-1])
        n = len(per_sentence)
        emb = np.zeros((n, self.max_tokens, dim), dtype=np.float16)
        mask = np.zeros((n, self.max_tokens), dtype=bool)  # True = posição válida

        truncated = 0
        for i, t in enumerate(per_sentence):
            arr = t.detach().cpu().float().numpy()
            length = min(arr.shape[0], self.max_tokens)
            if arr.shape[0] > self.max_tokens:
                truncated += 1
            emb[i, :length] = arr[:length].astype(np.float16)
            mask[i, :length] = True

        lengths = mask.sum(axis=1)
        print(f"Tokens por amostra: min={lengths.min()} mediana={int(np.median(lengths))} "
              f"max={lengths.max()} | truncadas: {truncated}/{n}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        np.savez_compressed(cache_path, emb=emb, mask=mask)
        print(f"Token embeddings salvos em {cache_path} {emb.shape}")
        return emb, mask

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

        # float16 no disco, float32 no treino.
        metadata = {
            "embeddings": torch.from_numpy(self.embeddings[idx].astype(np.float32)),
            "mask": torch.from_numpy(self.masks[idx].copy()),
        }
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_name, image, metadata, label

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

        # targets alinhado com cat.codes (ordem alfabética), não com unique().
        self.targets = list(metadata['diagnostic'].astype('category').cat.categories)
        if self.is_to_drop_nan is True:
            metadata = metadata.dropna().reset_index(drop=True)
        return metadata