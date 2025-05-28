from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
import pickle
import cv2

class SkinLesionDataset(Dataset):
    def __init__(self, metadata_file, img_dir, size=(224,224), drop_nan=False, 
                 bert_model_name='bert-base-uncased', random_undersampling=False, 
                 image_encoder="resnet-18", is_train=True):
        # Store parameters
        self.metadata_file = metadata_file
        self.img_dir = img_dir
        self.size = size
        self.is_to_drop_nan = drop_nan
        self.bert_model_name = bert_model_name
        self.random_undersampling = random_undersampling
        self.image_encoder = image_encoder
        self.is_train = is_train

        self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = self.load_transforms()

        # Load metadata and process
        self.metadata = self.load_metadata()
        self.features, self.labels, self.targets = self.one_hot_encoding()


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.metadata.iloc[idx]['img_id'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']

        metadata = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, metadata, label


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
        else:
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2()
            ])

    def load_metadata(self):
        # Carregar o CSV
        metadata = pd.read_csv(self.metadata_file).fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").\
           replace("NÃO  ENCONTRADO", "EMPTY").replace("BRASIL","BRAZIL")
        # Verificar se deve descartar linhas com NaN
        if self.is_to_drop_nan:
            metadata = metadata.dropna().reset_index(drop=True)

        return metadata
    
    def split_dataset(self, dataset, batch_size, test_size):
       # Dividir os índices do dataset
        indices = list(range(len(dataset)))
        train_indices, val_test_indices = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

        # Criar Subconjuntos
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_test_indices)

        # Criar DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def one_hot_encoding(self):
        # Seleção das features
        dataset_features = self.metadata.drop(columns=['patient_id', 'lesion_id', 'img_id', 'biopsed', 'diagnostic'])
        # Convert specific columns to numeric if possible
        # Definir as colunas categóricas e numéricas corretamente
        for col in ['age', 'diameter_1', 'diameter_2', 'fitspatrick']:
            dataset_features[col] = pd.to_numeric(dataset_features[col], errors='coerce')

        # Identify categorical and numerical columns
        categorical_cols = dataset_features.select_dtypes(include=['object', 'bool']).columns
        numerical_cols = dataset_features.select_dtypes(include=['float64', 'int64']).columns
        # Converter categóricas para string
        dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)

        # Preencher valores faltantes nas colunas numéricas com a média da coluna
        dataset_features[numerical_cols] = dataset_features[numerical_cols].fillna(-1)

        os.makedirs('./src/results/preprocess_data', exist_ok=True)

        # OneHotEncoder
        if os.path.exists("./src/results/preprocess_data/ohe.pickle"):
            with open('./src/results/preprocess_data/ohe.pickle', 'rb') as f:
                ohe = pickle.load(f)
            categorical_data = ohe.transform(dataset_features[categorical_cols])
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            categorical_data = ohe.fit_transform(dataset_features[categorical_cols])
            with open('./src/results/preprocess_data/ohe.pickle', 'wb') as f:
                pickle.dump(ohe, f)

        # StandardScaler
        if os.path.exists("./src/results/preprocess_data/scaler.pickle"):
            with open('./src/results/preprocess_data/scaler.pickle', 'rb') as f:
                scaler = pickle.load(f)
            numerical_data = scaler.transform(dataset_features[numerical_cols])
        else:
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(dataset_features[numerical_cols])
            with open('./src/results/preprocess_data/scaler.pickle', 'wb') as f:
                pickle.dump(scaler, f)

        # Concatenar dados
        processed_data = np.hstack((categorical_data, numerical_data))

        # Labels
        labels = self.metadata['diagnostic'].values
        if os.path.exists("./src/results/preprocess_data/label_encoder.pickle"):
            with open('./src/results/preprocess_data/label_encoder.pickle', 'rb') as f:
                label_encoder = pickle.load(f)
            encoded_labels = label_encoder.transform(labels)
        else:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            with open('./src/results/preprocess_data/label_encoder.pickle', 'wb') as f:
                pickle.dump(label_encoder, f)
                
        return processed_data, encoded_labels, self.metadata['diagnostic'].unique()

