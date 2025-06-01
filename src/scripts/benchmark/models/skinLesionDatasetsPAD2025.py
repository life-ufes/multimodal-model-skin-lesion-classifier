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
    def __init__(self, metadata_file, img_dir, drop_nan=False, bert_model_name='bert-base-uncased', size=(224, 224), is_train=False, random_undersampling=False, image_encoder="resnet-18"):
        # Inicializar argumentos
        self.metadata_file = metadata_file
        self.img_dir = img_dir
        self.size = size
        self.is_to_drop_nan = drop_nan
        self.bert_model_name = bert_model_name
        self.random_undersampling = random_undersampling
        self.image_encoder = image_encoder
        self.is_train = is_train
        self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.image_type = "CLINICAL" #"DERMATOSCOPE" # "CLINICAL" # Tipo de imagens
        self.transform = self.load_transforms()

        self.CLUSTER_TARGETS = {
            "C43": "MEL",
            "D03": "MEL",
            "D22": "NEVO",
            "C80": "CBC",
            "C44": "CEC",
            "D04": "CEC",
            "L57": "ACT",
            "L78": "NEVO",
            "L82": "SEBO",
        }

        # Carregar e processar metadados
        self.metadata = self.load_metadata()

        # Configuração de One-Hot Encoding para os metadados
        self.features, self.labels, self.targets = self.one_hot_encoding()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Carregar a imagem
        image_name = self.metadata.iloc[idx]['img-id']+".png"
        img_path = f"{self.img_dir}/{image_name}"

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = np.array(image)
        except Exception as e:
            print(f"[Erro] Não foi possível abrir imagem com PIL: {img_path} — {e}")
            raise FileNotFoundError(f"Imagem inválida: {img_path}")

        if self.transform:
            image = self.transform(image=image)['image']

        # Metadados e rótulo
        metadata = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_name, image, metadata, label

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
        metadata = pd.read_csv(self.metadata_file)
        
        # Substituir valores ausentes (NaN) nas colunas categóricas
        metadata["age"].fillna(0, inplace=False)
        metadata["age"] = metadata["age"].replace("EMPTY", 0)
        
        # Substituir valores específicos nas colunas de string
        replacements = {
            "EMPTY": "EMPTY", 
            " ": "EMPTY", 
            "  ": "EMPTY", 
            "NÃO  ENCONTRADO": "EMPTY", 
            "BRASIL": "BRAZIL", 
            "NAO PREENCHIDO": "EMPTY", 
            "I": "EMPTY"
        }

        # Aplicando substituições nas colunas categóricas (strings)
        metadata.replace(replacements, inplace=False)
        
        # Agora, preenchemos NaN nas colunas que podem ter valores ausentes
        # Aplique .fillna() de forma mais geral para garantir que outros NaN sejam substituídos, especialmente nas colunas de strings
        metadata.fillna("EMPTY", inplace=False)

        # Se o parâmetro is_to_drop_nan for True, remove qualquer linha com valores NaN
        if self.is_to_drop_nan:
            metadata.dropna(inplace=False)

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

    def convert_ids_labels(self):
        try:
            # Substituir os IDs pelos labels mapeados conforme o dicionário CLUSTER_TARGETS
            self.metadata['macroCIDDiagnostic'] = self.metadata['macroCIDDiagnostic'].map(self.CLUSTER_TARGETS)
            self.metadata = self.metadata.dropna(subset=['macroCIDDiagnostic'], inplace=False)
            dataframe_one_hot = pd.DataFrame(self.metadata)
            dataframe_one_hot.to_csv(f"./teste_{self.image_type}")
        
        except Exception as e:
            print(f"Erro ao converter os IDs para rótulos: {e}")
            return None


    def one_hot_encoding(self):
        # Converter IDs para labels
        self.convert_ids_labels()
        
        # Filtrar os dados para 'img-src' == 'CLINICAL'
        clinical_metadata = self.metadata[self.metadata['img-src'] == self.image_type].copy()
        
        # **Atualizar o self.metadata** para manter apenas as linhas filtradas
        self.metadata = clinical_metadata
        
        # Extraindo features e labels de forma consistente
        labels = clinical_metadata['macroCIDDiagnostic'].values
        dataset_features = clinical_metadata.drop(columns=["macroCIDDiagnostic"])
        # Selecionar as variáveis a serem pré-processadas
        categorical_cols = [
            "usePesticide", "gender", "familySkinCancerHistory", "familyCancerHistory", 
            "fitzpatrickSkinType", "macroBodyRegion", "hasItched", "hasGrown", "hasHurt", 
            "hasChanged", "hasBled", "hasElevation"
        ]
        numerical_cols = ["age"]
        
        # Converter variáveis categóricas para string
        dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)
        
        # Preencher valores faltantes nas colunas numéricas
        dataset_features[numerical_cols] = dataset_features[numerical_cols].fillna(-1)
        
        os.makedirs('./src/results/preprocess_data', exist_ok=True)
        
        # Carregar ou criar o OneHotEncoder
        ohe_file = './src/results/preprocess_data/ohe_pad_25.pickle'
        if os.path.exists(ohe_file):
            with open(ohe_file, 'rb') as f:
                ohe = pickle.load(f)
            categorical_data = ohe.transform(dataset_features[categorical_cols])
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            categorical_data = ohe.fit_transform(dataset_features[categorical_cols])
            with open(ohe_file, 'wb') as f:
                pickle.dump(ohe, f)
        
        # Carregar ou criar o StandardScaler
        scaler_file = './src/results/preprocess_data/scaler_pad_25.pickle'
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:  
                scaler = pickle.load(f)
            numerical_data = scaler.transform(dataset_features[numerical_cols])
        else:
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(dataset_features[numerical_cols])
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Concatenar dados pré-processados
        processed_data = np.hstack((categorical_data, numerical_data))

        # Codificação de Labels (target)
        label_encoder_file = './src/results/preprocess_data/label_encoder_pad_25.pickle'
        if os.path.exists(label_encoder_file):
            with open(label_encoder_file, 'rb') as f:
                label_encoder = pickle.load(f)
            encoded_labels = label_encoder.transform(labels)
        else:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            with open(label_encoder_file, 'wb') as f:
                pickle.dump(label_encoder, f)
        
        print(f"Shape dos processed_data {processed_data.shape}")
        print(f"Shape dos labels {len(labels)}")
        
        return processed_data, encoded_labels, clinical_metadata['macroCIDDiagnostic'].unique()
