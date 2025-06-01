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
from transformers import AutoTokenizer
import os
import pickle
import cv2

class SkinLesionDataset(Dataset):
    def __init__(self, metadata_file, img_dir, drop_nan=False, bert_model_name='bert-base-uncased', size=(224, 224), is_train=False, image_encoder="resnet-18", random_undersampling=False):
        # Inicializar argumentos
        self.metadata_file = metadata_file
        self.is_to_drop_nan = drop_nan
        self.img_dir = img_dir
        self.image_encoder = image_encoder
        self.size = size
        self.is_train = is_train
        self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.targets = None
        self.transform = self.load_transforms()
        
        # Carregar e configurar o tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        
        # Se não houver pad_token, defina o eos_token como pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"  # Opcional, ajusta se necessário
        
        # Carregar e processar metadados
        self.metadata = self.load_metadata()

        # Codificar os rótulos
        self.labels = self.metadata['diagnostic'].astype('category').cat.codes

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_name = self.metadata.iloc[idx]['img_id']
        img_path = os.path.abspath(os.path.join(self.img_dir, image_name))

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = np.array(image)
        except Exception as e:
            print(f"[Erro] Não foi possível abrir imagem com PIL: {img_path} — {e}")
            raise FileNotFoundError(f"Imagem inválida: {img_path}")

        if self.transform:
            image = self.transform(image=image)['image']


        # Processar texto dos metadados
        textual_data = self.metadata.iloc[idx]['sentence']
        
        # Tokenização do texto (garantindo que o modelo do GPT-2 ou BERT seja tratado corretamente)
        tokenized_text = self.tokenizer(
            textual_data,
            padding='max_length',         # Preenche até o tamanho máximo
            truncation=True,              # Trunca se exceder o limite
            max_length=256,               # Define o tamanho máximo da sequência
            return_tensors="pt"
        )
        
        # Rótulo
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_name, image, tokenized_text, label

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
        metadata = pd.read_csv(self.metadata_file).fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").replace("NÃO  ENCONTRADO", "EMPTY").replace("BRASIL", "BRAZIL")
        
        # Obter as classes
        self.targets = metadata['diagnostic'].unique()
        # Verificar se deve descartar linhas com NaN
        if self.is_to_drop_nan is True:
            metadata = metadata.dropna().reset_index(drop=True)
       
        return metadata

    def split_dataset(self, batch_size, test_size=0.2, random_state=42):
        # Dividir os índices do dataset
        indices = list(range(len(self)))
        train_indices, val_test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Criar DataLoaders
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_test_sampler = torch.utils.data.SubsetRandomSampler(val_test_indices)

        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        val_test_loader = DataLoader(self, batch_size=batch_size, sampler=val_test_sampler)

        return train_loader, val_test_loader