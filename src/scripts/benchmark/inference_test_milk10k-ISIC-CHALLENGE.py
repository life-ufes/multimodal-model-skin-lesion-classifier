import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import multimodalIntraInterModal
from models import skinLesionDatasetsMILK10K 

def load_model(model, device, model_path):
    try:
        model.to(device)
        model.eval() # Essencial para inferência!
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Carregamento bem sucedido do modelo!\n")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        raise SystemError("Erro ao carregar o modelo")
    return model

def encode_test_metadata(metadata, encoders):
    """
    Processa apenas metadados do dataset de teste usando encoders do treino.
    """
    categorical_cols = encoders["ohe"].feature_names_in_
    numerical_cols = encoders["scaler"].feature_names_in_

    cat_data = metadata[categorical_cols].astype(str).fillna("EMPTY")
    num_data = metadata[numerical_cols].fillna(-1).astype(float)

    cat_encoded = encoders["ohe"].transform(cat_data)
    num_scaled = encoders["scaler"].transform(num_data)

    features = np.hstack([cat_encoded, num_scaled]).astype(np.float32)
    return features


if __name__ == "__main__":
    # === Hiperparâmetros e Configurações ===
    num_classes = 11
    num_heads = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "davit_tiny.msft_in1k"
    text_model_encoder = "one-hot-encoder"
    common_dim = 512
    attention_mecanism = "att-intramodal+residual+cross-attention-metadados"
    batch_size = 8 # Definir batch_size para inferência
    # === Salvar Resultados no formato One-Hot ===
    target_names = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF',
                    'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
    # Peso do modelo usado
    weight_fold_path = "./src/results/MILK10k/unfrozen_weights/8/att-intramodal+residual+cross-attention-metadados/model_davit_tiny.msft_in1k_with_one-hot-encoder_512_with_best_architecture/davit_tiny.msft_in1k_fold_1/best-model/best_model.pt"
    # Predições das amostras do dataset de teste
    output_path = "./data/MILK10k_Test_Predictions.csv"

    print(f"Usando dispositivo: {device}")

    # === Carregar encoders usados no treino ===
    try:
        with open("./src/results/preprocess_data/ohe_milk10k_clinical: close-up.pickle", "rb") as f:
            ohe = pickle.load(f)
        with open("./src/results/preprocess_data/scaler_milk10k_clinical: close-up.pickle", "rb") as f:
            scaler = pickle.load(f)
        encoders = {"ohe": ohe, "scaler": scaler}
        print("Encoders carregados com sucesso.")
    except FileNotFoundError as e:
        raise SystemExit(f"Erro: Arquivo de encoder não encontrado. Verifique os caminhos. Detalhe: {e}")

    num_metadata_features = len(ohe.get_feature_names_out()) + scaler.mean_.shape[0]

    # === Criar a instância do modelo ===
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes, num_heads, device,
        cnn_model_name=model_name,
        text_model_name=text_model_encoder,
        common_dim=common_dim,
        vocab_size=num_metadata_features,
        unfreeze_weights=False, # Geralmente False para inferência
        attention_mecanism=attention_mecanism,
        n=2
    )

    # === Carregar os pesos do modelo treinado ===
    loaded_model = load_model(model=model, device=device, model_path=weight_fold_path)

    # === Criar o Dataset de Teste ===
    test_dataset = skinLesionDatasetsMILK10K.SkinLesionDataset(
        metadata_file="./data/MILK10k/MILK10k_Test_Metadata.csv",
        train_ground_truth=None,  # No teste não há labels
        img_dir="./data/MILK10k/MILK10k_Test_Input",
        size=(224, 224),
        image_type="clinical: close-up",
        is_train=False
    )

    # Processar os metadados com os encoders do treino
    test_dataset.features = encode_test_metadata(test_dataset.metadata, encoders)

        
    # === Criar o DataLoader para o Teste ===
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # === Loop de Inferência ===
    image_names_list = []
    predictions_list = []
    images = []

    with torch.no_grad():
        for (image_names, images, metadata, label) in tqdm(test_loader, desc="Realizando Inferência"):
            images = images.to(device)
            metadata = metadata.to(device)

            outputs = loaded_model(images, metadata)  # logits
            _, predicted_indices = torch.max(outputs, 1)

            image_names_list.extend(image_names)
            predictions_list.extend(predicted_indices.cpu().numpy())

    # Cria matriz one-hot
    one_hot_preds = np.zeros((len(predictions_list), len(target_names)))
    for i, pred_idx in enumerate(predictions_list):
        one_hot_preds[i, pred_idx] = 1.0

    # Monta DataFrame final
    df_predictions = pd.DataFrame(one_hot_preds, columns=target_names)
    df_predictions.insert(0, 'lesion_id', image_names_list)

    df_predictions.to_csv(output_path, index=False)

    print(f"\nPredições salvas com sucesso em: {output_path}")
