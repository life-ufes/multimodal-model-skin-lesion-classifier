import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import model_metrics, save_predictions
from utils.early_stopping import EarlyStopping
from utils import load_local_variables
import models.focalLoss as focalLoss
from models import multimodalIntraInterModal, dynamicMultimodalmodel, controllerMultimodalmodel
from models import skinLesionDatasets, skinLesionDatasetsWithBert
from utils.save_model_and_metrics import save_model_and_metrics
from collections import Counter
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import DataLoader, Subset
import numpy as np
import mlflow
from tqdm import tqdm

# Função para calcular os pesos das classes garantindo que haja um peso para cada classe
def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    weights = []
    for i in range(num_classes):
        if counts[i] > 0:
            weight = total_samples / (num_classes * counts[i])
        else:
            weight = 0.0
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float)

def train_process(num_epochs, 
                  num_heads, 
                  fold_num, 
                  train_loader, 
                  val_loader, 
                  targets, 
                  model, 
                  device, 
                  weightes_per_category, 
                  common_dim, 
                  model_name, 
                  text_model_encoder, 
                  attention_mecanism, 
                  results_folder_path):

    criterion = nn.CrossEntropyLoss(weight=weightes_per_category)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        verbose=True
    )
    model.to(device)

    model_save_path = os.path.join(
        results_folder_path, 
        f"model_{model_name}_with_{text_model_encoder}_{common_dim}_with_best_architecture"
    )
    os.makedirs(model_save_path, exist_ok=True)
    print(model_save_path)

    early_stopping = EarlyStopping(
        patience=10, 
        delta=0.01, 
        verbose=True,
        path=str(model_save_path + f'/step_{str(fold_num)}/best-model/'),
        save_to_disk=True,
        early_stopping_metric_name="val_loss"
    )

    initial_time = time.time()
    epoch_index = 0

    experiment_name = f"EXPERIMENTOS-{dataset_folder_name}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=(
            f"image_extractor_model_{model_name}_with_mecanism_"
            f"{attention_mecanism}_step_{fold_num}_num_heads_{num_heads}"
        )
    ):
        mlflow.log_param("step", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mecanism", attention_mecanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", "cross_entropy")
        mlflow.log_param("num_heads", num_heads)

        # Loop de treinamento
        for epoch_index in range(num_epochs):
            model.train()
            running_loss = 0.0

            for batch_index, ( _, image, metadata, label) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch_index+1}/{num_epochs}", leave=False)):
                image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            print(f"\nTraining: Epoch {epoch_index}, Loss: {train_loss:.4f}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ , image, metadata, label in val_loader:
                    # print(f"Image names: {image_name}\n")
                    image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            scheduler.step(val_loss)
            current_lr = [pg['lr'] for pg in optimizer.param_groups]
            print(f"Current Learning Rate(s): {current_lr}\n")

            metrics, all_labels, all_predictions = model_metrics.evaluate_model(
                model=model, dataloader = val_loader, device=device, fold_num=fold_num, targets=targets, base_dir=model_save_path 
            )
            metrics["epoch"] = epoch_index
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)
            print(f"Metrics: {metrics}")

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=epoch_index + 1)
                else:
                    mlflow.log_param(metric_name, metric_value)

            early_stopping(val_loss=val_loss, val_bacc=float(metrics["balanced_accuracy"]), model=model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
    
    train_process_time = time.time() - initial_time
    
    # Carrega o melhor modelo encontrado
    model = early_stopping.load_best_weights(model)
    model.eval()
    # Inferência para validação com o melhor modelo
    with torch.no_grad():
        metrics, all_labels, all_predictions = model_metrics.evaluate_model(
            model=model, dataloader = val_loader, device=device, fold_num=fold_num, targets=targets, base_dir=model_save_path 
        )

    metrics["train process time"] = str(train_process_time)
    metrics["epochs"] = str(int(epoch_index))
    metrics["data_val"] = "val"

    save_model_and_metrics(
        model=model, 
        metrics=metrics, 
        model_name=model_name, 
        base_dir=model_save_path,
        save_to_disk=True, 
        fold_num=fold_num, 
        all_labels=all_labels, 
        all_predictions=all_predictions, 
        targets=targets, 
        data_val="val"
    )
    print(f"Model saved at {model_save_path}")

    return model, model_save_path, metrics


def pipeline(dataset, num_metadata_features, num_epochs, k_folds, batch_size, device, num_classes, model_name, 
             num_heads, common_dim, text_model_encoder, unfreeze_weights, attention_mecanism, 
             results_folder_path, SEARCH_STEPS, search_space, num_workers=10, persistent_workers=True, 
             test_size=0.2):  # test_size é a proporção da validação
             
    labels = [dataset.labels[i] for i in range(len(dataset))]
    
    # Split simples com estratificação
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=labels,
        random_state=42
    )
    
    print("Usando split simples (train/val)")

    # Criar dataset de treino
    train_dataset = type(dataset)(
        metadata_file=dataset.metadata_file,
        img_dir=dataset.img_dir,
        size=(224,224),
        drop_nan=dataset.is_to_drop_nan,
        bert_model_name=dataset.bert_model_name,
        image_encoder=dataset.image_encoder,
        is_train=True  # Ativa augmentações de treino
    )
    train_dataset.metadata = dataset.metadata.iloc[train_idx].reset_index(drop=True)
    train_dataset.features, train_dataset.labels, train_dataset.targets = train_dataset.one_hot_encoding()

    # Criar dataset de validação
    val_dataset = type(dataset)(
        metadata_file=dataset.metadata_file,
        img_dir=dataset.img_dir,
        size=(224,224),
        drop_nan=dataset.is_to_drop_nan,
        bert_model_name=dataset.bert_model_name,
        image_encoder=dataset.image_encoder,
        is_train=False  # Transforms de validação
    )
    val_dataset.metadata = dataset.metadata.iloc[val_idx].reset_index(drop=True)
    val_dataset.features, val_dataset.labels, val_dataset.targets = val_dataset.one_hot_encoding()

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)

    train_labels = [labels[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    print(f"Pesos das classes: {class_weights}")
    
    controller = controllerMultimodalmodel.Controller(search_space).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=5e-4)
    baseline = None

    best_reward = 0.0
    best_config = None

    for step in range(1, SEARCH_STEPS + 1):
        config, log_prob = controller.sample_config()
        
        try:
            dynamic_model = dynamicMultimodalmodel.DynamicCNN(
                config, num_classes=num_classes, device=device,
                common_dim=common_dim, num_heads=num_heads, vocab_size=num_metadata_features,
                attention_mecanism=attention_mecanism, 
                n=1 if attention_mecanism=="no-metadata" else 2
            )

            dynamic_model, model_save_path, metrics = train_process(
                num_epochs=num_epochs, num_heads=num_heads, fold_num=step, train_loader=train_loader, val_loader=val_loader, 
                targets=dataset.targets, model=dynamic_model, device=device, weightes_per_category=class_weights, 
                common_dim=common_dim, model_name=model_name, text_model_encoder=text_model_encoder, attention_mecanism=attention_mecanism, results_folder_path=results_folder_path
            )

            save_predictions.model_val_predictions(
                model=dynamic_model, dataloader=val_loader, device=device,
                fold_num=step, targets=dataset.targets, base_dir=model_save_path
            )

            reward = metrics["balanced_accuracy"]

        except Exception as e:
            print(f"Erro no modelo com config {config}: {e}")
            reward = 0.0

        if reward > best_reward:
            best_reward = reward
            best_config = config
            print(f"🎉 Nova melhor arquitetura encontrada! Reward: {best_reward:.4f}")

        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline

        loss = -log_prob * advantage
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[{step}/{SEARCH_STEPS}] Reward: {reward:.4f} | Baseline: {baseline:.4f} | Config: {config}")

    print("\n--- Busca Finalizada ---")
    print(f"Melhor Reward: {best_reward:.4f}")
    print(f"Melhor Arquitetura: {best_config}")

def run_expirements(dataset_folder_path:str, results_folder_path:str, llm_model_name_sequence_generator:str, num_epochs:int, batch_size:int, k_folds:int, common_dim:int, text_model_encoder:str, unfreeze_weights: bool, device, list_num_heads: list, list_of_attention_mecanism:list, list_of_models: list, SEARCH_STEPS, search_space):
    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    if (text_model_encoder in ['one-hot-encoder', "tab-transformer"]):
                        dataset = skinLesionDatasets.SkinLesionDataset(
                        metadata_file=f"{dataset_folder_path}/metadata.csv",
                        img_dir=f"{dataset_folder_path}/images",
                        bert_model_name=text_model_encoder,
                        image_encoder=model_name,
                        drop_nan=False,
                        size=(224,224))
                    elif (text_model_encoder in ['gpt2', 'bert-base-uncased']):
                        dataset = skinLesionDatasetsWithBert.SkinLesionDataset(
                        metadata_file=f"{dataset_folder_path}/metadata_with_sentences_new-prompt-{llm_model_name_sequence_generator}.csv",
                        img_dir=f"{dataset_folder_path}/images",
                        bert_model_name=text_model_encoder,
                        image_encoder=model_name,
                        drop_nan=False,
                        size=(224,224))
                    else:
                        raise ValueError("Encoder de texto não implementado!\n")
                    
                    num_metadata_features = dataset.features.shape[1] if text_model_encoder == 'one-hot-encoder' else 512
                    print(f"Número de features do metadados: {num_metadata_features}\n")
                    num_classes = len(dataset.metadata['diagnostic'].unique())

                    pipeline(dataset, 
                        num_metadata_features=num_metadata_features, 
                        num_epochs=num_epochs, batch_size=batch_size, 
                        device=device, k_folds=-1, num_classes=num_classes, 
                        model_name=model_name, common_dim=common_dim, 
                        text_model_encoder=text_model_encoder,
                        num_heads=num_heads,
                        unfreeze_weights=unfreeze_weights,
                        attention_mecanism=attention_mecanism, 
                        results_folder_path=f"{results_folder_path}/{num_heads}/{attention_mecanism}",
                        SEARCH_STEPS = SEARCH_STEPS, 
                        search_space = search_space,
                        num_workers=5, persistent_workers=True
                    )
                except Exception as e:
                    print(f"Erro ao processar o treino do modelo {model_name} e com o mecanismo: {attention_mecanism}. Erro:{e}\n")
                    continue

if __name__ == "__main__":
    # Carrega os dados localmente
    local_variables = load_local_variables.get_env_variables()
    num_epochs =  1 ## Treino com poucas épocas # local_variables["num_epochs"]
    batch_size = local_variables["batch_size"]
    k_folds = 1 ## Treino com poucas épocas # local_variables["k_folds"]
    common_dim = local_variables["common_dim"]
    list_num_heads = local_variables["list_num_heads"]
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = local_variables["dataset_folder_path"]
    unfreeze_weights = bool(local_variables["unfreeze_weights"])
    llm_model_name_sequence_generator = local_variables["llm_model_name_sequence_generator"]
    results_folder_path = local_variables["results_folder_path"]
    results_folder_path = f"{results_folder_path}/{dataset_folder_name}/{'unfrozen_weights' if unfreeze_weights else 'frozen_weights'}"
    # Métricas para o experimento
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model_encoder = 'one-hot-encoder' # "tab-transformer" # 'bert-base-uncased' # 'gpt2' # 'one-hot-encoder'
    # Para todas os tipos de estratégias a serem usadas
    list_of_attention_mecanism = ["att-intramodal+residual+cross-attention-metadados"] # ["att-intramodal+residual", "att-intramodal+residual+cross-attention-metadados", "att-intramodal+residual+cross-attention-metadados+att-intramodal+residual", "weighted-after-crossattention", "cross-weights-after-crossattention", "crossattention", "concatenation", "no-metadata", "weighted", "metablock"] # ["att-intramodal+residual+cross-attention-metadados"] # ["att-intramodal+residual", "att-intramodal+residual+cross-attention-metadados", "att-intramodal+residual+cross-attention-metadados+att-intramodal+residual", "weighted-after-crossattention", "cross-weights-after-crossattention", "crossattention", "concatenation", "no-metadata", "weighted", "metablock"]
    # Testar com todos os modelos
    list_of_models = ["custom-cnn-with-NAS"] # ["nextvit_small.bd_ssld_6m_in1k", "mvitv2_small.fb_in1k", "coat_lite_small.in1k","davit_tiny.msft_in1k", "caformer_b36.sail_in22k_ft_in1k", "beitv2_large_patch16_224.in1k_ft_in22k_in1k", "vgg16", "mobilenet-v2", "densenet169", "resnet-50"]
    
    # Treina todos modelos que podem ser usados no modelo multi-modal
    search_space = {
        "num_blocks": [2, 3, 4],                        # Número de blocos convolucionais
        "initial_filters": [16, 32, 64],                # Filtros no primeiro bloco
        "kernel_size": [3, 5],                          # Tamanho do Kernel para todas as convs
        "layers_per_block": [1, 2],                     # Camadas conv por bloco
        "use_pooling": [True, False]                   # Usar MaxPool após cada bloco
    }

    SEARCH_STEPS = 1

    run_expirements(
        dataset_folder_path, 
        results_folder_path,
        llm_model_name_sequence_generator, 
        num_epochs, 
        batch_size, 
        k_folds, 
        common_dim, 
        text_model_encoder, 
        unfreeze_weights, 
        device, 
        list_num_heads, 
        list_of_attention_mecanism=list_of_attention_mecanism, 
        list_of_models=list_of_models,
        SEARCH_STEPS= SEARCH_STEPS, 
        search_space= search_space
    )
