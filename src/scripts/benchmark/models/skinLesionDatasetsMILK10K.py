from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
import pickle
import cv2


class SkinLesionDataset(Dataset):
    def __init__(
        self,
        metadata_file,
        train_ground_truth,
        img_dir,
        drop_nan=False,
        bert_model_name='bert-base-uncased',
        size=(224, 224),
        is_train=False,
        random_undersampling=False,
        image_encoder="resnet-18",
        type_of_problem="multiclass",
        image_type="clinical: close-up"
    ):
        # argumentos principais
        self.image_type = image_type
        self.metadata_file = metadata_file
        self.train_ground_truth = train_ground_truth
        self.is_to_drop_nan = drop_nan
        self.img_dir = img_dir
        self.image_encoder = image_encoder
        self.bert_model_name = bert_model_name
        self.random_undersampling = random_undersampling
        self.size = size
        self.is_train = is_train
        self.type_of_problem = type_of_problem
        self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = self.load_transforms()

        # Carregar metadados + ground truth
        self.metadata = self.load_metadata()
        # Se for treino (tem ground truth), roda o one_hot_encoding normal
        if self.train_ground_truth is not None:
            self.features, self.labels, self.targets = self.one_hot_encoding()
        else:
            # No teste, apenas inicializa vazio
            self.features, self.labels, self.targets = None, None, None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_name = self.metadata.iloc[idx]['isic_id']
        lesion_id_name = self.metadata.iloc[idx]['lesion_id']
        img_path = os.path.join(self.img_dir, str(lesion_id_name), f"{image_name}.jpg")

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = np.array(image)
        except Exception as e:
            print(f"[Erro] Não foi possível abrir imagem com PIL: {img_path} — {e}")
            raise FileNotFoundError(f"Imagem inválida: {img_path}")

        # Albumentations retorna dict com a chave 'image'
        if self.transform:
            image = self.transform(image=image)['image']

        metadata = torch.tensor(self.features[idx], dtype=torch.float32)
        label = "None"
        
        # Se for treino/validação:
        if (self.train_ground_truth is not None):
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Dataset de teste, não tem labels
        return lesion_id_name, image, metadata, label

    def load_transforms(self):
        if self.is_train:
            drop_prob = float(np.random.uniform(0.0, 0.05))
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
                        max_holes=1,
                        max_height=4,
                        max_width=4,
                        min_holes=1,
                        min_height=4,
                        min_width=4,
                        p=1
                    ),
                ], p=0.1),
                A.OneOf([
                    A.OneOrOther(
                        first=A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, p=1),
                        second=A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=False, p=1),
                        p=0.5
                    ),
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

    def load_metadata(self):
        # Carregar CSV de metadados
        metadata = pd.read_csv(self.metadata_file, dtype=str)
        metadata = metadata.fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").replace("NÃO  ENCONTRADO", "EMPTY")

        # Filtrar pelo tipo de imagem ANTES do merge
        metadata = metadata[metadata['image_type'] == self.image_type].reset_index(drop=True)

        # Carregar ground truth
        if self.train_ground_truth is None:
            print(f"Arquivo de ground truth não encontrado: {self.train_ground_truth}")
            pass
        else:
            df_groundtruth = pd.read_csv(self.train_ground_truth, dtype=str)

            # Merge pelo lesion_id
            metadata = metadata.merge(df_groundtruth, on='lesion_id', how='left', suffixes=('', '_gt'))

        if self.is_to_drop_nan:
            metadata = metadata.dropna().reset_index(drop=True)

        return metadata.reset_index(drop=True)

    def one_hot_encoding(self):
        # Colunas categóricas e numéricas
        drop_cols = ['image_type', 'attribution', 'copyright_license']

        # Colunas categóricas de fato
        categorical_cols = [
            'image_manipulation', 'sex', 'skin_tone_class', 'site'
        ]

        # Colunas numéricas
        numerical_cols = [
            'age_approx',
            'MONET_ulceration_crust',
            'MONET_hair',
            'MONET_vasculature_vessels',
            'MONET_erythema',
            'MONET_pigmented',
            'MONET_gel_water_drop_fluid_dermoscopy_liquid',
            'MONET_skin_markings_pen_ink_purple_pen'
        ]

        # Descarta colunas administrativas
        self.metadata = self.metadata.drop(columns=drop_cols, errors="ignore")

        # Garantir colunas existentes
        for c in categorical_cols + numerical_cols:
            if c not in self.metadata.columns:
                self.metadata[c] = np.nan

        # Preprocessamento
        self.metadata['age_approx'] = pd.to_numeric(self.metadata['age_approx'], errors='coerce')
        self.metadata[categorical_cols] = self.metadata[categorical_cols].astype(str).fillna("EMPTY")
        self.metadata[numerical_cols] = self.metadata[numerical_cols].fillna(-1).astype(float)

        os.makedirs('./src/results/preprocess_data', exist_ok=True)

        # OneHotEncoder para categóricas
        ohe_path = f"./src/results/preprocess_data/ohe_milk10k_{self.image_type}.pickle"
        if os.path.exists(ohe_path):
            with open(ohe_path, 'rb') as f:
                ohe = pickle.load(f)
            categorical_data = ohe.transform(self.metadata[categorical_cols])
        else:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            categorical_data = ohe.fit_transform(self.metadata[categorical_cols])
            with open(ohe_path, 'wb') as f:
                pickle.dump(ohe, f)

        # StandardScaler para numéricas
        scaler_path = f"./src/results/preprocess_data/scaler_milk10k_{self.image_type}.pickle"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            numerical_data = scaler.transform(self.metadata[numerical_cols])
        else:
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(self.metadata[numerical_cols])
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        processed_data = np.hstack((categorical_data, numerical_data)).astype(np.float32)

        # ⚠️ Colunas de diagnóstico = todas do ground truth
        diagnosis_cols = [c for c in self.metadata.columns if c not in (
            ['lesion_id', 'isic_id', 'image_type'] + categorical_cols + numerical_cols
        )]

        if len(diagnosis_cols) == 0:
            raise KeyError("Nenhuma coluna de diagnóstico encontrada no ground truth!")

        # Labels (argmax no one-hot)
        y_onehot = self.metadata[diagnosis_cols].astype(float).values
        encoded_labels = np.argmax(y_onehot, axis=1)

        # Lista de classes
        targets = diagnosis_cols

        return processed_data, encoded_labels.astype(np.int64), targets
