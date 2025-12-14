import torch
import torch.nn as nn
import os
import csv
import sys
from pydantic import ValidationError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import model_metrics, save_predictions
from utils.early_stopping import EarlyStopping
from utils import load_local_variables
from models.pydantic_llm_response_formats import NASConfig
import models.focalLoss as focalLoss
from models import multimodalIntraInterModal, dynamicMultimodalmodel
from models import skinLesionDatasets, skinLesionDatasetsWithBert
from utils.request_to_llm import request_to_ollama, filter_generated_response
from utils.save_model_and_metrics import save_model_and_metrics
from collections import Counter
from sklearn.model_selection import train_test_split
import time
import json
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

def train_process(config:dict, num_epochs:int, 
                  num_heads:int, 
                  fold_num:int, 
                  train_loader, 
                  val_loader, 
                  targets, 
                  model, 
                  device:str, 
                  weightes_per_category:str, 
                  common_dim:str, 
                  model_name:str, 
                  text_model_encoder, 
                  attention_mecanism:str, 
                  llm_model_name:str,
                  results_folder_path:str):

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
        save_to_disk=False,
        early_stopping_metric_name="val_bacc"
    )

    initial_time = time.time()
    epoch_index = 0

    # usa variável global dataset_folder_name definida no main
    experiment_name = f"EXPERIMENTOS-NAS-{dataset_folder_name} -- LLM AS CONTROLLER WITH TRAIN PROCESS HISTORY AND PYDANTIC - OPTIMIZATION TRAIN PROCESS - 13/12/2025"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=(
            f"image_extractor_model_{model_name}_with_mechanism_"
            f"{attention_mecanism}_step_{fold_num}_num_heads_{num_heads}"
        ), nested=True
    ):
        mlflow.log_param("step", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mechanism", attention_mecanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", "cross_entropy")
        mlflow.log_param("num_heads", num_heads)
        mlflow.log_param("controller_llm_model_name", llm_model_name)

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
                model=model, dataloader = val_loader, device=device, fold_num=fold_num, targets=targets, base_dir=model_save_path, model_name=model_name
            )
            metrics["epoch"] = epoch_index
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)
            metrics["attention_mechanism"] = str(attention_mecanism)
            metrics["common_dim"]=int(common_dim)

            print(f"Metrics: {metrics}\n")

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
            model=model, dataloader = val_loader, device=device, fold_num=fold_num, targets=targets, base_dir=model_save_path, model_name=model_name 
        )
    
    metrics["train process time"] = str(train_process_time)
    metrics["epochs"] = str(int(epoch_index))
    metrics["data_val"] = "val"
    metrics["epoch"] = epoch_index
    metrics["train_loss"] = float(train_loss)
    metrics["val_loss"] = float(val_loss)
    metrics["attention_mechanism"] = str(attention_mecanism)
    metrics["common_dim"]=int(common_dim)

    print(f"Model saved at {model_save_path}")
    
    # Salvar os dados da configuração
    folder_name = f"{model_name}_fold_{fold_num}"
    folder_path = os.path.join(model_save_path, folder_name)

    # Criar a pasta para o modelo
    os.makedirs(folder_path, exist_ok=True)

    save_model_and_metrics(
        model=model, 
        metrics=metrics, 
        model_name=model_name, 
        base_dir=folder_path,
        save_to_disk=False, 
        fold_num=fold_num, 
        all_labels=all_labels, 
        all_predictions=all_predictions, 
        targets=targets, 
        data_val="val"
    )

    # Salvar as métricas
    metrics_file = os.path.join(results_folder_path, "all_model_metrics.csv")
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    
    with open(os.path.join(folder_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return model, model_save_path, metrics

def pipeline(
    dataset,
    num_metadata_features,
    num_epochs,
    batch_size,
    device,
    num_classes,
    model_name,
    num_heads,
    common_dim,
    k_folds,
    text_model_encoder,
    unfreeze_weights,
    attention_mecanism,
    results_folder_path,
    SEARCH_STEPS,
    search_space,
    num_workers=5,
    persistent_workers=True,
    test_size=0.2,
    llm_model_name_sequence_generator: str = "qwen3:0.6b"
):

    # ============================================================
    # Split estratificado
    # ============================================================
    labels = [dataset.labels[i] for i in range(len(dataset))]
    os.makedirs(results_folder_path, exist_ok=True)

    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=labels,
        random_state=42
    )

    # ============================================================
    # Dataset de treino
    # ============================================================
    train_dataset = type(dataset)(
        metadata_file=dataset.metadata_file,
        img_dir=dataset.img_dir,
        size=(224, 224),
        drop_nan=dataset.is_to_drop_nan,
        bert_model_name=dataset.bert_model_name,
        image_encoder=dataset.image_encoder,
        is_train=True
    )
    train_dataset.metadata = dataset.metadata.iloc[train_idx].reset_index(drop=True)
    train_dataset.features, train_dataset.labels, train_dataset.targets = (
        train_dataset.one_hot_encoding()
    )

    # ============================================================
    # Dataset de validação
    # ============================================================
    val_dataset = type(dataset)(
        metadata_file=dataset.metadata_file,
        img_dir=dataset.img_dir,
        size=(224, 224),
        drop_nan=dataset.is_to_drop_nan,
        bert_model_name=dataset.bert_model_name,
        image_encoder=dataset.image_encoder,
        is_train=False
    )
    val_dataset.metadata = dataset.metadata.iloc[val_idx].reset_index(drop=True)
    val_dataset.features, val_dataset.labels, val_dataset.targets = (
        val_dataset.one_hot_encoding()
    )
    val_targets = val_dataset.targets

    # ============================================================
    # Dataloaders
    # ============================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )

    # ============================================================
    # Pesos de classe
    # ============================================================
    train_labels = [labels[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels, num_classes).to(device)

    print(f"Pesos das classes: {class_weights}")

    # ============================================================
    # NAS Controller
    # ============================================================
    history = []
    tested_configs = set()
    best_reward = -float("inf")
    reward = -float("inf")
    config_llm = {}
    val_loss = 0.0
    best_config = None
    best_step = -1
    baseline = None

    llm_model_name = llm_model_name_sequence_generator

    with mlflow.start_run(nested=True):
        mlflow.log_param("controller_type", "LLM")
        mlflow.log_param("controller_llm_model_name", llm_model_name)
        mlflow.log_param("search_steps", SEARCH_STEPS)
        mlflow.log_param("search_space_json", json.dumps(search_space))

        for step in range(1, SEARCH_STEPS + 1):
            print(f"Step: {step}")

            history_text = (
                "No history yet."
                if not history
                else "\n".join(
                    f"Step {i+1}: BACC={h['reward']:.4f}, config={json.dumps(h['config'])}"
                    for i, h in enumerate(history)
                )
            )

            prompt = f"""
                You are an AI NAS controller for skin lesion classification.

                Goal: maximize Balanced Accuracy (BACC).

                Search space:
                {json.dumps(search_space, indent=2)}

                History:
                {history_text}

                Based on this history, propose ONE new architecture configuration that is likely to IMPROVE BACC compared to the best so far.
                You should:
                - Prefer configurations similar to high-BACC ones, but with small changes (tweak 1–3 hyperparameters at a time).
                - Avoid repeating exactly the same configuration as past attempts.
                - Always keep hyperparameter values inside the given search_space.
                - If the history is bad (low BACC), you may explore more diverse configurations.

                Return ONLY ONE JSON OBJECT (not a list), with EXACTLY these keys:
                {{
                "num_blocks": <int>,
                "initial_filters": <int>,
                "kernel_size": <int>,
                "layers_per_block": <int>,
                "use_pooling": <true or false>,
                "common_dim": <int>,
                "attention_mechanism": <string>,
                "num_layers_text_fc": <int>,
                "neurons_per_layer_size_of_text_fc": <int>,
                "num_layers_fc_module": <int>,
                "neurons_per_layer_size_of_fc_module": <int>
                }}
                
            """

            # ====================================================
            # LLM → JSON → PYDANTIC
            # ====================================================
            try:
                response = request_to_ollama(prompt, model_name=llm_model_name, host="http://localhost:11434", thinking=True, timeout=300)
                
                if response is not None:
                    raw_json = filter_generated_response(generated_sentence=response)
                    parsed = json.loads(raw_json)

                    config_obj = NASConfig.model_validate(parsed)
                    config_llm = config_obj.model_dump()

                    print(f"[Step {step}] Config válida: {config_llm}\n")
                else:
                    print(f"[Step {step}] Resposta do LLM inválida! Resposta do LLM:{response}\n")
                    continue

            except (json.JSONDecodeError, ValidationError) as e:
                print(f"[Step {step}] Config inválida descartada:")
                print(e)
                continue

            # Evita repetir arquitetura
            cfg_key = json.dumps(config_llm, sort_keys=True)
            if cfg_key in tested_configs:
                continue
            tested_configs.add(cfg_key)

            attention_cfg = config_llm["attention_mechanism"]
            common_dim_cfg = int(config_llm["common_dim"])

            # ====================================================
            # Instancia e treina modelo
            # ====================================================
            try:
                dynamic_model = dynamicMultimodalmodel.DynamicCNN(
                    config_llm,
                    num_classes=num_classes,
                    device=device,
                    common_dim=common_dim_cfg,
                    num_heads=num_heads,
                    vocab_size=num_metadata_features,
                    attention_mecanism=attention_cfg,
                    n=1 if attention_cfg == "no-metadata" else 2
                )

                dynamic_model, _, metrics = train_process(
                    config=config_llm,
                    num_epochs=num_epochs,
                    num_heads=num_heads,
                    fold_num=step,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    targets=val_targets,
                    model=dynamic_model,
                    device=device,
                    weightes_per_category=class_weights,
                    common_dim=common_dim_cfg,
                    model_name=model_name,
                    text_model_encoder=text_model_encoder,
                    attention_mecanism=attention_cfg,
                    llm_model_name=llm_model_name,
                    results_folder_path=results_folder_path
                )

                reward = float(metrics["balanced_accuracy"])
                val_loss = float(metrics["val_loss"])

            except Exception as e:
                print(f"[Step {step}] Erro ao treinar arquitetura:")
                print(e)
                continue

        # ================================
        # Atualizações do controller
        # ================================
        if reward > best_reward:
            best_reward = reward
            best_config = config_llm
            best_step = step
            print(f"🏆 Nova melhor arquitetura! BACC={best_reward:.4f} (step {best_step})")

        history.append({"config": config_llm, "reward": reward})

        # Mantém apenas Top-K
        history = sorted(history, key=lambda x: x["reward"], reverse=True)[:10]

        # Logging seguro
        mlflow.log_metric("controller_reward", reward, step=step)
        mlflow.log_metric("dynamic_cnn_val_loss", val_loss, step=step)
        mlflow.log_param(f"config_step_{step}", json.dumps(config_llm))


        # ====================================================
        # Final
        # ====================================================
        print("\n--- NAS FINALIZADO ---")
        print(f"Melhor BACC: {best_reward:.4f}")
        print(f"Step: {best_step}")
        print(f"Arquitetura: {best_config}")

        mlflow.log_metric(
            "final_best_reward",
            best_reward if best_reward != -float("inf") else 0.0,
            step=SEARCH_STEPS
        )

        if best_config is not None:
            mlflow.log_param(
                "final_best_architecture_config",
                json.dumps(best_config)
            )

        best_config_path = os.path.join(results_folder_path, "best_config.json")
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2)

        print(f"Best config salva em: {best_config_path}")

def run_expirements(dataset_folder_path:str, results_folder_path:str, llm_model_name_sequence_generator:str,
                    num_epochs:int, batch_size:int, k_folds:int, common_dim:int, text_model_encoder:str,
                    unfreeze_weights: bool, device, list_num_heads: list, list_of_attention_mecanism:list,
                    list_of_models: list, SEARCH_STEPS, search_space):

    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    if text_model_encoder in ['one-hot-encoder', "tab-transformer"]:
                        dataset = skinLesionDatasets.SkinLesionDataset(
                            metadata_file=f"{dataset_folder_path}/metadata.csv",
                            img_dir=f"{dataset_folder_path}/images",
                            bert_model_name=text_model_encoder,
                            image_encoder=model_name,
                            drop_nan=False,
                            size=(224,224)
                        )
                    elif text_model_encoder in ['gpt2', 'bert-base-uncased']:
                        dataset = skinLesionDatasetsWithBert.SkinLesionDataset(
                            metadata_file=f"{dataset_folder_path}/metadata_with_sentences_new-prompt-{llm_model_name_sequence_generator}.csv",
                            img_dir=f"{dataset_folder_path}/images",
                            bert_model_name=text_model_encoder,
                            image_encoder=model_name,
                            drop_nan=False,
                            size=(224,224)
                        )
                    else:
                        raise ValueError("Encoder de texto não implementado!\n")
                    
                    num_metadata_features = dataset.features.shape[1] if text_model_encoder == 'one-hot-encoder' else 512
                    print(f"Número de features do metadados: {num_metadata_features}\n")
                    num_classes = len(dataset.metadata['diagnostic'].unique())

                    # path específico deste experimento/modelo/mecanismo
                    current_results_path = f"{results_folder_path}/{num_heads}/{attention_mecanism}"
                    os.makedirs(current_results_path, exist_ok=True)

                    pipeline(
                        dataset, 
                        num_metadata_features=num_metadata_features, 
                        num_epochs=num_epochs,
                        batch_size=batch_size, 
                        device=device,
                        k_folds=-1,
                        num_classes=num_classes, 
                        model_name=model_name,
                        common_dim=common_dim, 
                        text_model_encoder=text_model_encoder,
                        num_heads=num_heads,
                        unfreeze_weights=unfreeze_weights,
                        attention_mecanism=attention_mecanism, 
                        results_folder_path=current_results_path,
                        SEARCH_STEPS=SEARCH_STEPS, 
                        search_space=search_space,
                        num_workers=6,
                        persistent_workers=True,
                        llm_model_name_sequence_generator=llm_model_name_sequence_generator
                    )
                except Exception as e:
                    print(f"Erro ao processar o treino do modelo {model_name} e com o mecanismo: {attention_mecanism}. Erro:{e}\n")
                    continue


if __name__ == "__main__":
    # Carrega os dados localmente
    local_variables = load_local_variables.get_env_variables()
    num_epochs = local_variables["num_epochs"]  # Treino com poucas épocas
    batch_size = local_variables["batch_size"]
    k_folds = 1  # Treino com poucas épocas
    common_dim = -1
    list_num_heads = local_variables["list_num_heads"]
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = local_variables["dataset_folder_path"]
    unfreeze_weights = bool(local_variables["unfreeze_weights"])
    llm_model_name_sequence_generator = local_variables["llm_model_name_sequence_generator"]
    results_folder_path = local_variables["results_folder_path"]
    results_folder_path = f"{results_folder_path}/controller-{llm_model_name_sequence_generator}/{dataset_folder_name}/{'unfrozen_weights' if unfreeze_weights else 'frozen_weights'}"

    # Métricas para o experimento
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model_encoder = 'one-hot-encoder'  # "tab-transformer" # 'bert-base-uncased' # 'gpt2'
    
    # Para todas os tipos de estratégias a serem usadas
    list_of_attention_mecanism = ["custom-attention-mechanism"]
    # Testar com todos os modelos
    list_of_models = ["custom-cnn-with-NAS"]
    
    # Treina todos modelos que podem ser usados no modelo multi-modal
    search_space = {
        "num_blocks": [2, 5, 10],
        "initial_filters": [16, 32, 64],
        "kernel_size": [3, 5],
        "layers_per_block": [1, 2],
        "use_pooling": [True, False],
        "common_dim": [64, 128, 256, 512],
        "attention_mechanism": ["no-metadata", "concatenation", "crossattention", "metablock"],
        "num_layers_text_fc": [1, 2, 3],
        "neurons_per_layer_size_of_text_fc": [64, 128, 256, 512],
        "num_layers_fc_module": [1, 2],
        "neurons_per_layer_size_of_fc_module": [256, 512]
    }

    SEARCH_STEPS = 500

    run_expirements(
        dataset_folder_path=dataset_folder_path, 
        results_folder_path=results_folder_path,
        llm_model_name_sequence_generator=llm_model_name_sequence_generator, 
        num_epochs=num_epochs, 
        batch_size=batch_size, 
        k_folds=k_folds, 
        common_dim=common_dim, 
        text_model_encoder=text_model_encoder, 
        unfreeze_weights=unfreeze_weights, 
        device=device, 
        list_num_heads=list_num_heads, 
        list_of_attention_mecanism=list_of_attention_mecanism, 
        list_of_models=list_of_models,
        SEARCH_STEPS=SEARCH_STEPS, 
        search_space=search_space
    )