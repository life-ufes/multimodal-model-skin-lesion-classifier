from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import os
import pickle

class SkinLesionDataset(Dataset):
    def __init__(self, metadata_file, img_dir, drop_nan=False, bert_model_name='bert-base-uncased', random_undersampling=False, image_encoder="resnet-18"):
        # Inicializar argumentos
        self.metadata_file = metadata_file
        self.is_to_drop_nan = drop_nan
        self.img_dir = img_dir
        self.image_encoder = image_encoder
        self.image_type = "" # 
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
        img_path = f"{self.img_dir}/{self.metadata.iloc[idx]['img-id']}.png"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Metadados e rótulo
        metadata = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, metadata, label

    def load_transforms(self):
        if self.image_encoder=="vit-base-patch16-224":
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
            # Label Encoding
            if os.path.exists("./src/results/preprocess_data/label_encoder_pad_25.pickle"):
                # Carregar o LabelEncoder salvo
                with open('./src/results/preprocess_data/label_encoder_pad_25.pickle', 'rb') as f:
                    label_encoder = pickle.load(f)
            else:
                # Fazer o fit com todos os labels únicos presentes nos dados
                label_encoder = LabelEncoder()
                label_encoder.fit(self.metadata['macroCIDDiagnostic'].unique())

                # Salvar o LabelEncoder
                with open('./src/results/preprocess_data/label_encoder_pad_25.pickle', 'wb') as f:
                    pickle.dump(label_encoder, f)

            # Convertendo os rótulos usando o LabelEncoder
            encoded_labels = label_encoder.transform(self.metadata['macroCIDDiagnostic'].values)

            return encoded_labels

        except Exception as e:
            print(f"Erro ao converter os IDs para rótulos: {e}")
            return None


    def one_hot_encoding(self):
        # Codificação dos dados
        self.convert_ids_labels()  # Converte os IDs para os labels necessários
        # Seleção das features
        dataset_features = self.metadata
        # dataset_features = dataset_features[dataset_features['img-src'] == 'CLINICAL']
        dataset_features=dataset_features.drop(columns=["macroCIDDiagnostic"])
        # Selecionar apenas as variáveis a serem pré-processadas
        categorical_cols = [
            "usePesticide", "gender", "familySkinCancerHistory", "familyCancerHistory", 
            "fitzpatrickSkinType", "macroBodyRegion", "hasItched", "hasGrown", "hasHurt", 
            "hasChanged", "hasBled", "hasElevation"
        ]
        numerical_cols = ["age"]

        # Converter categóricas para string
        dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)
        
        # Preencher valores faltantes nas colunas numéricas com -1 (ou média, se preferir)
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

        # Concatenar dados
        processed_data = np.hstack((categorical_data, numerical_data))

        print(processed_data.shape)

        # Codificação de Labels (target)
        labels = self.metadata['macroCIDDiagnostic'].values
        
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
        return processed_data, encoded_labels, self.metadata['macroCIDDiagnostic'].unique()
