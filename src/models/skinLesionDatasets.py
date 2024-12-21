from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch

class SkinLesionDataset(Dataset):
    def __init__(self, metadata_file, img_dir, transform=None):
        # Carregar metadados e inicializar transformações
        self.metadata = pd.read_csv(metadata_file)
        self.img_dir = img_dir
        self.transform = self.load_transforms()

        # Configuração de One-Hot Encoding para os metadados
        self.encoder = OneHotEncoder(sparse=False)
        self.features = self.encoder.fit_transform(self.metadata.drop(columns=['diagnostic', 'img_id']))

        # Codificar os rótulos
        self.labels = self.metadata['diagnostic'].astype('category').cat.codes

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Carregar a imagem
        img_path = f"{self.img_dir}/{self.metadata.iloc[idx]['img_id']}"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Metadados e rótulo
        metadata = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, metadata, label

    def load_transforms(self):
        # Transforma imagens para o formato necessário para treinamento
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform
