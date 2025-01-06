import os
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import pickle
from models import multimodalIntraInterModal  # Ajuste conforme a estrutura do seu projeto


def one_hot_encoding(metadata):
    # Remover colunas desnecessárias e selecionar as features
    dataset_features = metadata.drop(columns=['patient_id', 'lesion_id', 'img_id', 'biopsed', 'diagnostic'])

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
    print(f"{dataset_features[numerical_cols]}\n")
    # Assegurar que as colunas categóricas usadas na inferência correspondem às usadas no treinamento
    dataset_features_categorical = dataset_features[categorical_cols]

    # OneHotEncoder
    ohe_path = "./src/results/preprocess_data/ohe.pickle"
    if os.path.exists(ohe_path):
        with open(ohe_path, 'rb') as f:
            ohe = pickle.load(f)
        categorical_data = ohe.transform(dataset_features_categorical)
    else:
        raise FileNotFoundError("Arquivo OneHotEncoder não encontrado. Certifique-se de que o modelo foi treinado corretamente.")

    # StandardScaler para dados numéricos
    scaler_path = "./src/results/preprocess_data/scaler.pickle"
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        numerical_data = scaler.transform(dataset_features[numerical_cols])
    else:
        raise FileNotFoundError("Arquivo StandardScaler não encontrado. Certifique-se de que o modelo foi treinado corretamente.")

    # Concatenar os dados processados
    processed_metadata = np.hstack((categorical_data, numerical_data))
    return processed_metadata


def load_transforms(image_encoder="densenet169"):
    if image_encoder == "vit-base-patch16-224":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


def process_image(img, image_encoder="densenet169"):
    image = img.convert("RGB")
    transform = load_transforms(image_encoder)
    return transform(image)


def load_multimodal_model(device, model_path):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=6,
        device=device,
        cnn_model_name="densenet169",
        text_model_name="one-hot-encoder",
        vocab_size=86,
        attention_mecanism="crossattention"
    )
    model.to(device)
    model.eval()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model


def inference(processed_image, processed_metadata, device, model_path):
    model = load_multimodal_model(device, model_path)
    model.eval()
    
    # Adiciona a dimensão de batch
    processed_image = processed_image.unsqueeze(0).to(device)
    processed_metadata = torch.tensor(processed_metadata, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(processed_image, processed_metadata)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    return predictions, probabilities


def process_data(text, column_names):
    # Dividir o texto em campos separados por vírgula
    data = pd.DataFrame([text.split(',')], columns=column_names)
    # Substituir valores vazios ou indesejados por "EMPTY"
    data=data.fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").\
           replace("NÃO  ENCONTRADO", "EMPTY")
    data = data.replace(r'^\s*$', 'EMPTY', regex=True)  # Substituir strings vazias ou espaços
    data = data.replace(['NÃO ENCONTRADO', 'n/a', None], 'EMPTY')  # Substituir indicadores comuns de valores vazios
    data = data.replace("BRASIL", "BRAZIL")
    return data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/after_finetunning/densenet169/crossattention/model_densenet169_with_one-hot-encoder_1024/densenet169_fold_1_20250105_131137/model.pth"

    # Carregar imagem de teste
    image = Image.open("/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/data/images/PAT_771_1491_390.png")
    processed_image = process_image(image, image_encoder="densenet169")

    # Definir nomes das colunas
    column_names = [
        "patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age",
        "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
        "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic", "itch",
        "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"
    ]

    # Carregar dados de teste
    text = "PAT_771,1491,True,True,ITALY,ITALY,69,False,MALE,False,True,True,True,3.0,FACE,6.0,3.0,BCC,True,UNK,False,UNK,False,True,PAT_771_1491_390.png,True"  # "PAT_1516,1765,,,,,8,,,,,,,,ARM,,,NEV,False,False,False,False,False,False,PAT_1516_1765_530.png,False"
    metadata = process_data(text, column_names)

    # Processar metadados
    processed_metadata = one_hot_encoding(metadata)
    print(f"Processed_metadata:{processed_metadata}\n")

    # Realizar inferência
    predictions, probabilities = inference(processed_image, processed_metadata, device, model_path)

    print(f"Predictions: {predictions}\n")
    print(f"Probabilities: {probabilities}\n")
