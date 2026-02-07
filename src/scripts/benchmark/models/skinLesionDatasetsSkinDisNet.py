from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
import pickle
import cv2

class SkinDisNetDataset(Dataset):

    def __init__(
        self,
        csv_file,
        img_root,
        size=(224,224),
        is_train=True,
        preprocess_dir="./data/preprocess_data/skindisnet",
        fit_encoders=False,
        build_features=True   # <<< NOVO
    ):
        self.csv_file = csv_file
        self.img_root = img_root
        self.size = size
        self.is_train = is_train
        self.preprocess_dir = preprocess_dir
        self.fit_encoders = fit_encoders
        self.build_features = build_features
        self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = self.load_transforms()

        self.metadata = self._load_metadata()

        # 🔥🔥🔥 PONTO CRÍTICO 🔥🔥🔥
        if self.build_features:
            self.features, self.labels, self.targets = self._process_metadata()
        else:
            # Dataset base: só metadata
            self.features = None
            self.labels = None
            self.targets = None

    def load_transforms(self):
        if self.is_train:
            drop_prob = np.random.uniform(0.0, 0.05)
            return A.Compose([
                A.Affine(scale={"x": (1.0, 1.25), "y": (1.0, 1.25)}, p=0.25),
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
        else:
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2()
            ])


    def _load_metadata(self):
        df = pd.read_csv(self.csv_file)
        # Fill NaN and strip whitespace
        df = df.fillna("UNKNOWN")
        return df

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Compose image path
        folder = row["Folder_name"]
        image_name = row["Image_id"]
        img_path = os.path.join(self.img_root, folder, image_name+".jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[SkinDisNet] Image not found: {img_path}")
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = np.array(image)
        except Exception as e:
            print(f"[Erro] Não foi possível abrir imagem com PIL: {img_path} — {e}")
            raise FileNotFoundError(f"Imagem inválida: {img_path}")

        if self.transform:
            image = self.transform(image=image)['image']

        meta = torch.tensor(self.features[idx], dtype=torch.float32)
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_name, image, meta, lbl

    def _process_metadata(self):

        df = self.metadata.copy()

        label_col = "Diagnosis"

        # Define numerical + categorical
        numerical_cols = ["Age"]
        categorical_cols = ["Sex", "Leision_location"]

        df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors="coerce").fillna(-1)
        df[categorical_cols] = df[categorical_cols].astype(str)

        feature_df = df[numerical_cols + categorical_cols]

        os.makedirs(self.preprocess_dir, exist_ok=True)

        ohe_path = os.path.join(self.preprocess_dir, "ohe_skdn.pkl")
        le_path  = os.path.join(self.preprocess_dir, "le_skdn.pkl")

        # OneHotEncoder
        if self.fit_encoders:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            Xcat = ohe.fit_transform(feature_df[categorical_cols])
            with open(ohe_path,"wb") as f: pickle.dump(ohe,f)
        else:
            with open(ohe_path,"rb") as f: ohe = pickle.load(f)
            Xcat = ohe.transform(feature_df[categorical_cols])

        Xnum = feature_df[numerical_cols].values
        X = np.hstack([Xcat, Xnum])

        # Labels
        labels = df[label_col].astype(str).values
        if self.fit_encoders:
            le = LabelEncoder()
            y = le.fit_transform(labels)
            with open(le_path,"wb") as f: pickle.dump(le,f)
        else:
            with open(le_path,"rb") as f: le = pickle.load(f)
            y = le.transform(labels)

        return X, y, le.classes_

    def _build_transforms(self):
        if self.is_train:
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2(),
            ])