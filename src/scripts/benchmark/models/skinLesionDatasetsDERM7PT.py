import os
import pickle
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2


class Derm7ptDataset(Dataset):
    """
    Dataset para o Derm7pt, retornando:
        - image_name (str)
        - image (Tensor CxHxW normalizado)
        - metadata (Tensor de features tabulares)
        - label (Tensor long)

    Assumindo CSV no formato com colunas:
    case_num, diagnosis, seven_point_score, pigment_network, ..., diagnosis_number, etc.
    """

    def __init__(
        self,
        metadata_file: str,
        img_dir: str,
        size: tuple = (224, 224),
        is_train: bool = True,
        is_to_drop_nan: bool = False,
        preprocess_dir: str = "./data/preprocess_data",
        image_type:str="derm" # "derm"
    ):
        self.metadata_file = metadata_file
        self.img_dir = img_dir
        self.image_type = image_type
        self.size = size
        self.is_train = is_train
        self.is_to_drop_nan = is_to_drop_nan
        self.preprocess_dir = preprocess_dir

        self.normalization = ([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
        self.transform = self._build_transforms()

        # Carrega e prepara metadados
        self.metadata = self._load_metadata()
        self.features, self.labels, self.targets = self._process_metadata()

    # -------------------------------------------------------------------------
    # Métodos obrigatórios do Dataset
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Nome da imagem dermatoscópica
        image_name = row[self.image_type]
        img_path = os.path.abspath(os.path.join(self.img_dir, image_name))

        # Carregar imagem
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = np.array(image)
        except Exception as e:
            print(f"[Erro] Não foi possível abrir imagem com PIL: {img_path} — {e}")
            raise FileNotFoundError(f"Imagem inválida: {img_path}")

        # Transformações de imagem
        if self.transform:
            image = self.transform(image=image)["image"]

        # Metadados e rótulo
        metadata_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_name, image, metadata_tensor, label_tensor

    # -------------------------------------------------------------------------
    # Transforms de imagem
    # -------------------------------------------------------------------------
    def _build_transforms(self):
        if self.is_train:
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),

                A.Rotate(
                    limit=45,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5
                ),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),

                A.GaussianBlur(sigma_limit=(0, 2.0), p=0.25),

                A.CoarseDropout(
                    max_holes=5,
                    max_height=8,
                    max_width=8,
                    p=0.15
                ),

                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.25
                ),
                A.RandomBrightnessContrast(p=0.25),

                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2(),
            ])

    # -------------------------------------------------------------------------
    # Carregamento e limpeza dos metadados
    # -------------------------------------------------------------------------
    def _load_metadata(self):
        metadata = pd.read_csv(self.metadata_file)

        # Opcional: normalizar nomes de colunas com espaços
        # Ex.: "location_genital areas" -> "location_genital_areas"
        metadata.columns = [
            c.strip().replace(" ", "_") for c in metadata.columns
        ]

        # Preencher NaNs básicos
        metadata = metadata.replace("NÃO  ENCONTRADO", "EMPTY")
        metadata = metadata.fillna("EMPTY")

        if self.is_to_drop_nan:
            metadata = metadata.replace("EMPTY", np.nan).dropna().reset_index(drop=True)

        return metadata

    # -------------------------------------------------------------------------
    # Processamento dos metadados (OneHot + StandardScaler + LabelEncoder)
    # -------------------------------------------------------------------------
    def _process_metadata(self):
        """
        Cria o vetor de features tabulares e os labels (diagnosis).

        - numéricas:
            seven_point_score,
            *_number (as scores clínicos)
        - categóricas:
            critérios categóricos, localização, sexo, etc.
        """

        # Colunas a ignorar no vetor de features
        ignore_cols = [
            "case_num",
            "case_id",
            "clinic",
            "derm",
            "notes",
            "split",
            "diagnosis",
            "diagnosis_number",
        ]

        # Colunas numéricas (scores do checklist)
        numerical_cols = [
            "seven_point_score",
            "pigment_network_number",
            "streaks_number",
            "pigmentation_number",
            "regression_structures_number",
            "dots_and_globules_number",
            "blue_whitish_veil_number",
            "vascular_structures_number",
        ]

        # Criar cópia para manipulação
        df = self.metadata.copy()

        # Garantir que colunas ausentes não quebrem o código
        for col in numerical_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Features = tudo menos as colunas de controle
        feature_df = df.drop(columns=[c for c in ignore_cols if c in df.columns])

        # As colunas numéricas precisam estar presentes em feature_df
        # (se tiver algo fora, fica como categórica)
        for col in numerical_cols:
            if col not in feature_df.columns:
                # Garante que está lá para não quebrar o índice
                feature_df[col] = np.nan

        # Separar listas finais
        numerical_cols_final = [c for c in numerical_cols if c in feature_df.columns]
        categorical_cols = [c for c in feature_df.columns if c not in numerical_cols_final]

        # Converter categóricas para string
        feature_df[categorical_cols] = feature_df[categorical_cols].astype(str)

        # Converter numéricas para float e preencher NaN com -1
        feature_df[numerical_cols_final] = feature_df[numerical_cols_final].apply(
            pd.to_numeric, errors="coerce"
        )
        feature_df[numerical_cols_final] = feature_df[numerical_cols_final].fillna(-1)

        # Diretório base para salvar/usar encoders
        os.makedirs(self.preprocess_dir, exist_ok=True)

        # -----------------------------
        # OneHotEncoder (categóricas)
        # -----------------------------
        ohe_path = os.path.join(self.preprocess_dir, "ohe_derm7pt.pickle")
        if os.path.exists(ohe_path):
            with open(ohe_path, "rb") as f:
                ohe = pickle.load(f)
            categorical_data = ohe.transform(feature_df[categorical_cols])
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            categorical_data = ohe.fit_transform(feature_df[categorical_cols])
            with open(ohe_path, "wb") as f:
                pickle.dump(ohe, f)

        # -----------------------------
        # StandardScaler (numéricas)
        # -----------------------------
        scaler_path = os.path.join(self.preprocess_dir, "scaler_derm7pt.pickle")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            numerical_data = scaler.transform(feature_df[numerical_cols_final])
        else:
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(feature_df[numerical_cols_final])
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        # Concatenar tudo
        processed_data = np.hstack((categorical_data, numerical_data))

        # -----------------------------
        # Labels (diagnosis)
        # -----------------------------
        labels = self.metadata["diagnosis"].values
        le_path = os.path.join(self.preprocess_dir, "label_encoder_derm7pt.pickle")
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                label_encoder = pickle.load(f)
            encoded_labels = label_encoder.transform(labels)
        else:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            with open(le_path, "wb") as f:
                pickle.dump(label_encoder, f)

        targets = np.unique(labels)

        return processed_data, encoded_labels, targets
