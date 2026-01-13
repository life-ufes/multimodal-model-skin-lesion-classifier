import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
import pickle
import cv2
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


class MIDASDataset(Dataset):

    def __init__(
        self,
        metadata_file,
        img_dir,
        size=(224,224),
        is_train=True,
        preprocess_dir="./data/preprocess_data/midas",
        fit_encoders=False,
        build_features=True   # <<< NOVO
    ):
        self.metadata_file = metadata_file
        self.img_dir = img_dir
        self.size = size
        self.is_train = is_train
        self.preprocess_dir = preprocess_dir
        self.fit_encoders = fit_encoders
        self.build_features = build_features
    
        self.normalization = ([0.485,0.456,0.406],[0.229,0.224,0.225])
        self.transform = self._build_transforms()

        self.metadata = self._load_metadata()

        if build_features:
            self.features, self.labels, self.targets = self._process_metadata()
        else:
            self.features = None
            self.labels = None
            self.targets = None


    # ------------------------------------------------
    # Metadata
    # ------------------------------------------------
    def _load_metadata(self):
        df = pd.read_excel(self.metadata_file)
        df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
        df = df.fillna("EMPTY")
        return df

    # ------------------------------------------------
    # Dataset API
    # ------------------------------------------------
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        raw_name = str(row["midas_file_name"]).strip()
        base = os.path.splitext(raw_name)[0]

        # Diretórios candidatos
        dirs = [self.img_dir]
        images_sub = os.path.join(self.img_dir, "images")
        if os.path.isdir(images_sub):
            dirs.append(images_sub)

        # Lista de caminhos candidatos
        candidates = []
        for d in dirs:
            candidates.append(os.path.join(d, raw_name))
            candidates.append(os.path.join(d, base + ".jpg"))
            candidates.append(os.path.join(d, base + ".jpeg"))

        img_path = None
        for cand in candidates:
            if os.path.exists(cand):
                img_path = cand
                break

        # Se não encontrar imagem, pula
        if img_path is None:
            return None

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            image = self.transform(image=image)["image"]
        except Exception:
            return None

        return (
            raw_name,
            image,
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    # ------------------------------------------------
    # Tabular processing (fold-aware)
    # ------------------------------------------------
    def _process_metadata(self):

        numerical_cols = ["midas_age", "length_mm", "width_mm"]

        categorical_cols = [
            "midas_gender", "midas_fitzpatrick", "midas_ethnicity", "midas_race",
            "midas_location", "midas_melanoma", "midas_distance",
            "clinical_impression_1", "clinical_impression_2", "clinical_impression_3"
        ]

        df = self.metadata.copy()

        # --------------------------------------------------
        # BINARY CLINICAL LABEL
        # --------------------------------------------------
        df["binary_label"] = (
            df["midas_path"]
            .astype(str)
            .str.lower()
            .str.startswith("malignant")
            .astype(int)
        )

        # --------------------------------------------------
        # NUMERICAL
        # --------------------------------------------------
        df[numerical_cols] = df[numerical_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(-1)

        # --------------------------------------------------
        # CATEGORICAL
        # --------------------------------------------------
        df[categorical_cols] = df[categorical_cols].astype(str)

        feature_df = df[numerical_cols + categorical_cols]

        os.makedirs(self.preprocess_dir, exist_ok=True)

        ohe_path = os.path.join(self.preprocess_dir, "ohe_midas.pkl")
        scaler_path = os.path.join(self.preprocess_dir, "scaler_midas.pkl")

        # -------------------------------
        # One-Hot Encoder
        # -------------------------------
        if self.fit_encoders:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            Xcat = ohe.fit_transform(feature_df[categorical_cols])
            with open(ohe_path, "wb") as f:
                pickle.dump(ohe, f)
        else:
            if not os.path.exists(ohe_path):
                raise FileNotFoundError(f"[MIDAS] OHE não encontrado em {ohe_path}")
            with open(ohe_path, "rb") as f:
                ohe = pickle.load(f)
            Xcat = ohe.transform(feature_df[categorical_cols])

        # -------------------------------
        # StandardScaler
        # -------------------------------
        if self.fit_encoders:
            scaler = StandardScaler()
            Xnum = scaler.fit_transform(feature_df[numerical_cols])
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        else:
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"[MIDAS] Scaler não encontrado em {scaler_path}")
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            Xnum = scaler.transform(feature_df[numerical_cols])

        # --------------------------------------------------
        # Final feature vector
        # --------------------------------------------------
        X = np.hstack([Xcat, Xnum])

        # --------------------------------------------------
        # Labels (already 0/1)
        # --------------------------------------------------
        y = df["binary_label"].values.astype(int)

        targets = np.array(["benign", "malignant"])

        return X, y, targets


    # ------------------------------------------------
    # Transforms
    # ------------------------------------------------
    def _build_transforms(self):
        if self.is_train:
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),
                A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.GaussianBlur(sigma_limit=(0, 2.0), p=0.25),
                A.CoarseDropout(
                    num_holes_range=(1, 5),
                    hole_height_range=(8, 8),
                    hole_width_range=(8, 8),
                    p=0.15
                ),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.25),
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
