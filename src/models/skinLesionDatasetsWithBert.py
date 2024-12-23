from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch

class SkinLesionDataset(Dataset):
    def __init__(self, metadata_file, img_dir, drop_nan=False, transform=None):
        # Inicializar argumentos
        self.metadata_file = metadata_file
        self.is_to_drop_nan = drop_nan
        self.img_dir = img_dir
        self.transform = self.load_transforms()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Carregar e processar metadados
        self.metadata = self.load_metadata()

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

        # Processar texto dos metadados
        textual_data = self.metadata.iloc[idx].drop(['patient_id', 'lesion_id', 'img_id', 'diagnostic']).fillna('')
        text = ' '.join(map(str, textual_data.values))
        tokenized_text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            max_length=200
        )

        # Rótulo
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, tokenized_text, label

    def load_transforms(self):
        # Transforma imagens para o formato necessário para treinamento
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def load_metadata(self):
        # Carregar o CSV
        metadata = pd.read_csv(self.metadata_file)

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