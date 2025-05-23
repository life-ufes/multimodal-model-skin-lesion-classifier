from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch

class SkinLesionDataset(Dataset):
    def __init__(self, metadata_file, img_dir, drop_nan=False, bert_model_name='bert-base-uncased', image_encoder="resnet-18", random_undersampling=False):
        # Inicializar argumentos
        self.metadata_file = metadata_file
        self.is_to_drop_nan = drop_nan
        self.img_dir = img_dir
        self.image_encoder = image_encoder
        self.random_undersampling = random_undersampling
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
        # Carregar a imagem
        img_path = f"{self.img_dir}/{self.metadata.iloc[idx]['img_id']}"
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
        
        if self.transform:
            image = self.transform(image)

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

        return image, tokenized_text, label

    def load_transforms(self):
        if self.image_encoder == "vit-base-patch16-224":
            # Transforma imagens para o formato necessário para treinamento
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(360),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Garantir que os valores estejam no intervalo [0, 1]
                transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(360),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform

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