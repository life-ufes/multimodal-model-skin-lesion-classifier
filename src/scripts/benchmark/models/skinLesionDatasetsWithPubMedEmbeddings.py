import os
import re
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from model2vec import StaticModel

class SkinLesionDataset(Dataset):
    RAW_TEXT_ENCODERS = [
        "pubmedbert-base-embeddings",
        "pubmedbert-base-embeddings-100K",
        "pubmedbert-base-embeddings-500K",
        "pubmedbert-base-embeddings-1M"
        "pubmedbert-base-embeddings-2M"
    ]

    def __init__(
        self,
        metadata_file,
        img_dir,
        drop_nan=False,
        bert_model_name='bert-base-uncased',
        image_encoder="resnet-18",
        random_undersampling=False
    ):
        # Inicializar argumentos
        self.metadata_file = metadata_file
        self.is_to_drop_nan = drop_nan
        self.img_dir = img_dir
        self.bert_model_name = bert_model_name.lower()
        self.image_encoder = image_encoder
        self.random_undersampling = random_undersampling

        # Flag para raw text (PubMedBERT embeddings)
        self.use_raw_text = any(
            self.bert_model_name.startswith(name)
            for name in self.RAW_TEXT_ENCODERS
        )

        # Inicializar tokenizer se precisar
        if self.bert_model_name in ["bert-base-uncased", "gpt2"]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            # Ajuste de pad_token para GPT2
            if 'gpt2' in self.bert_model_name and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
        else:
            self.tokenizer = StaticModel.from_pretrained(f"neuml/{self.bert_model_name}")

        # Preparar transforms de imagem
        self.transform = self.load_transforms()

        # Carregar e processar metadados
        self.metadata = self.load_metadata()

        # Codificar rótulos e armazenar mapeamentos
        categories = self.metadata['diagnostic'].astype('category').cat.categories
        self.label2id = {cat: idx for idx, cat in enumerate(categories)}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.labels = self.metadata['diagnostic'].map(self.label2id).tolist()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Carregar imagem
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.img_dir, row['img_id'])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
        if self.transform:
            image = self.transform(image)

        # Processar texto
        textual_data = row['sentence'] if isinstance(row['sentence'], str) else ""
        if self.use_raw_text:
            text_input = self.tokenizer.encode(textual_data)
        else:
            tokenized = self.tokenizer(
                textual_data,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            text_input = {
                'input_ids': tokenized['input_ids'].squeeze(0),
                'attention_mask': tokenized['attention_mask'].squeeze(0)
            }

        # Rótulo
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, text_input, label

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
        metadata = pd.read_csv(self.metadata_file)
        if (self.bert_model_name=="one-hot-encoder"):
            metadata=metadata.fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").replace("NÃO  ENCONTRADO", "EMPTY").replace("BRASIL", "BRAZIL")
        
        # Obter as classes
        self.targets = metadata['diagnostic'].unique()
        # Verificar se deve descartar linhas com NaN
        if self.is_to_drop_nan is True:
            metadata = metadata.dropna().reset_index(drop=True)
       
        return metadata