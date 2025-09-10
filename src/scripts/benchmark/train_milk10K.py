import torch
import torch.nn as nn
from utils import model_metrics, save_predictions
from utils.early_stopping import EarlyStopping
import models.focalLoss as focalLoss
from models import multimodalIntraInterModal
from models import skinLesionDatasetsMILK10K
from torchvision.transforms import v2
from utils import load_local_variables
from utils.save_model_and_metrics import save_model_and_metrics
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import numpy as np
from collections import Counter
import time
import os
from torch.utils.data import DataLoader, Subset
# Importações do MLflow
import mlflow
from tqdm import tqdm


def compute_class_weights(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    return torch.tensor([class_weights[cls] for cls in sorted(class_counts.keys())], dtype=torch.float)


def train_process(num_epochs, 
                  num_heads, 
                  fold_num, 
                  train_loader, 
                  val_loader, 
                  targets, 
                  num_classes,
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

    # Uso de CutMix e MixUP 
    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    # ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        verbose=True
    )
    model.to(device)

    # Save the final (or best) model
    model_save_path = os.path.join(
        results_folder_path, 
        f"model_{model_name}_with_{text_model_encoder}_{common_dim}_with_best_architecture"
    )

    os.makedirs(model_save_path, exist_ok=True)
    print(model_save_path)

    # Instantiate EarlyStopping
    early_stopping = EarlyStopping(
        patience=10, 
        delta=0.00, 
        verbose=True,
        path=str(model_save_path + f'/{model_name}_fold_{fold_num}/best-model/'),
        save_to_disk=True,
        early_stopping_metric_name="val_bacc"
    )

    initial_time = time.time()
    epoch_index = 0  # Track the epoch

    # Set your MLflow experiment
    experiment_name = "EXPERIMENTOS-MILK10K - NEW GATED ATTENTION BASED AND RESIDUAL BLOCK"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=(
            f"image_extractor_model_{model_name}_with_mecanism_"
            f"{attention_mecanism}_fold_{fold_num}_num_heads_{num_heads}"
        )
    ):
        # Log MLflow parameters
        mlflow.log_param("fold_num", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mecanism", attention_mecanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", "cross_entropy")
        mlflow.log_param("num_heads", num_heads)

        # -----------------------------
        # Training Loop
        # -----------------------------
        for epoch_index in range(num_epochs):
            model.train()
            running_loss = 0.0

            # Adicionando barra de progresso para o loop de batches
            for batch_index, (_, image, metadata, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch_index+1}/{num_epochs}", leave=False)):
                image, metadata, label = (image.to(device), metadata.to(device),label.to(device))
                image, label = cutmix_or_mixup(image, label)

                optimizer.zero_grad()
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            print(f"\nTraining: Epoch {epoch_index}, Loss: {train_loss:.4f}")

            # -----------------------------
            # Validation Loop
            # -----------------------------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (_, image, metadata, label) in val_loader:
                    image, metadata, label = (
                        image.to(device),
                        metadata.to(device),
                        label.to(device)
                    )
                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Step the scheduler with validation loss
            scheduler.step(val_loss)
            current_lr = [pg['lr'] for pg in optimizer.param_groups]
            print(f"Current Learning Rate(s): {current_lr}\n")

            # -----------------------------
            # Evaluate Metrics
            # -----------------------------
            metrics, all_labels, all_predictions = model_metrics.evaluate_model(
                model=model, dataloader = val_loader, device=device, fold_num=fold_num, targets=targets, base_dir=model_save_path, model_name=model_name 
            )
            metrics["epoch"] = epoch_index
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)
            print(f"Metrics: {metrics}")

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=epoch_index + 1)
                else:
                    mlflow.log_param(metric_name, metric_value)

            # -----------------------------
            # Early Stopping
            # -----------------------------
            early_stopping(val_loss=val_loss, val_bacc=float(metrics["balanced_accuracy"]), model=model)

            # Check if we should stop early
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    # End of training
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

    return model, model_save_path


def pipeline(dataset, num_metadata_features, num_epochs, batch_size, device, k_folds, num_classes, model_name, num_heads, common_dim, text_model_encoder, unfreeze_weights, attention_mecanism, results_folder_path, num_workers=10, persistent_workers=True):
    # Obter os rótulos para validação estratificada (se necessário)
    labels = [dataset.labels[i] for i in range(len(dataset))]
    targets = dataset.targets
    # Configurar o StratifiedKFold
    stratifiedKFold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(stratifiedKFold.split(range(len(dataset)), labels)):
        print(f"Fold {fold+1}/{k_folds}")

        train_dataset = type(dataset)(
            metadata_file=dataset.metadata_file,
            img_dir=dataset.img_dir,
            size=dataset.size,
            drop_nan=dataset.is_to_drop_nan,
            train_ground_truth=dataset.train_ground_truth,
            image_type= dataset.image_type,
            bert_model_name=dataset.bert_model_name,
            image_encoder=dataset.image_encoder
        )
        

        val_dataset = type(dataset)(
            metadata_file=dataset.metadata_file,
            img_dir=dataset.img_dir,
            size=dataset.size,
            drop_nan = dataset.is_to_drop_nan,
            train_ground_truth =dataset.train_ground_truth,
            image_type = dataset.image_type,
            bert_model_name=dataset.bert_model_name,
            image_encoder=dataset.image_encoder  
        )
        
        # train_dataset.metadata = dataset.metadata.iloc[train_idx].reset_index(drop=True)
        # train_dataset.features, train_dataset.labels, train_dataset.targets = train_dataset.one_hot_encoding()

        # val_dataset.metadata = dataset.metadata.iloc[val_idx].reset_index(drop=True)
        # val_dataset.features, val_dataset.labels, val_dataset.targets = val_dataset.one_hot_encoding()

        # Criar DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)

        # Calcular pesos das classes com base no conjunto de treino
        train_labels = [labels[i] for i in train_idx]
        class_weights = compute_class_weights(train_labels).to(device)
        print(f"Pesos das classes no fold {fold+1}: {class_weights}")

        # Criar o modelo
        model = multimodalIntraInterModal.MultimodalModel(num_classes, num_heads, device, cnn_model_name=model_name, text_model_name=text_model_encoder, common_dim=common_dim, vocab_size=num_metadata_features, unfreeze_weights=unfreeze_weights, attention_mecanism=attention_mecanism, n=1 if attention_mecanism=="no-metadata" else 2)
        # Treinar o modelo no fold atual
        model, model_save_path = train_process(
            num_epochs=num_epochs, num_heads=num_heads, fold_num=fold+1, num_classes=num_classes, train_loader=train_loader, val_loader=val_loader, targets=targets, model=model, device=device,
            weightes_per_category=class_weights, common_dim=common_dim, model_name=model_name, text_model_encoder=text_model_encoder, attention_mecanism=attention_mecanism, results_folder_path=results_folder_path)
        # Salvar as predições em um arquivo csv
        save_predictions.model_val_predictions(model=model, dataloader=val_loader, device=device, fold_num=fold+1, targets= dataset.targets, base_dir=model_save_path)    

def run_expirements(dataset_folder_path:str, results_folder_path:str, num_epochs:int, num_workers:int, persistent_workers:bool, type_of_problem:str, image_type:str, batch_size:int, k_folds:int, common_dim:int, text_model_encoder:str, unfreeze_weights: bool, device, list_num_heads: list, list_of_attention_mecanism:list, list_of_models: list):
    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    dataset = skinLesionDatasetsMILK10K.SkinLesionDataset(
                    metadata_file=f"{dataset_folder_path}/MILK10k_Training_Metadata.csv",
                    train_ground_truth=f"{dataset_folder_path}/MILK10k_Training_GroundTruth.csv", # Inclui o 'lesion_id' e a sinalização do possível diagnostico/nome da classe (em one-hot encode)
                    img_dir=f"{dataset_folder_path}/MILK10k_Training_Input",
                    bert_model_name=text_model_encoder,
                    image_encoder=model_name,
                    drop_nan=False,
                    random_undersampling=False,
                    size=(224,224),
                    image_type=image_type)

                    num_metadata_features = dataset.features.shape[1] if text_model_encoder== 'one-hot-encoder' else 512
                    
                    print(f"Número de features do metadados: {num_metadata_features}\n")
                    print(f"Classes presentes: {dataset.targets}\n")
                    num_classes = len(dataset.targets)
                    print(f"Número de classes: {num_classes}\n")
                    pipeline(dataset, 
                        num_metadata_features=num_metadata_features, 
                        num_epochs=num_epochs, batch_size=batch_size, 
                        device=device, k_folds=k_folds, 
                        num_classes=num_classes, 
                        model_name=model_name, common_dim=common_dim, 
                        text_model_encoder=text_model_encoder,
                        num_heads=num_heads,
                        unfreeze_weights=unfreeze_weights,
                        attention_mecanism=attention_mecanism, 
                        results_folder_path=f"{results_folder_path}/{num_heads}/{attention_mecanism}", num_workers=num_workers, persistent_workers=True
                    )
                except Exception as e:
                    print(f"Erro ao processar o treino do modelo {model_name} e com o mecanismo: {attention_mecanism}. Erro:{e}\n")
                    continue

if __name__ == "__main__":
    # Carrega os dados localmente
    local_variables = load_local_variables.get_env_variables()
    num_epochs = int(local_variables["num_epochs"])
    batch_size = int(local_variables["batch_size"])
    k_folds = int(local_variables["k_folds"])
    common_dim = int(local_variables["common_dim"])
    num_workers=int(local_variables["num_workers"])    
    list_num_heads = local_variables["list_num_heads"]
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path= local_variables["dataset_folder_path"]
    unfreeze_weights = bool(local_variables["unfreeze_weights"]) # Caso queira descongelar os pesos da CNN desejada

    text_model_encoder = 'one-hot-encoder' #  'bert-base-uncased' # 'one-hot-encoder' # 'tab-transformer'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model_name_sequence_generator=local_variables["llm_model_name_sequence_generator"]
    type_of_problem = "multiclass" #"binaryclass" #"multiclass"
    image_type="clinical: close-up"
    results_folder_path = local_variables["results_folder_path"]
    results_folder_path = f"{results_folder_path}/{dataset_folder_name}/{'unfrozen_weights' if unfreeze_weights else 'frozen_weights'}"
    # Para todas os tipos de estratégias a serem usadas
    list_of_attention_mecanism = ["att-intramodal+residual+cross-attention-metadados"] # ["concatenation", "no-metadata", "att-intramodal+residual", "att-intramodal+residual+cross-attention-metadados", "att-intramodal+residual+cross-attention-metadados+att-intramodal+residual"] # ["gfcam", "cross-weights-after-crossattention", "crossattention", "concatenation", "no-metadata", "weighted"]
    # Testar com todos os modelos
    list_of_models = ["davit_tiny.msft_in1k"]# ["davit_tiny.msft_in1k", "mvitv2_small.fb_in1k", "coat_lite_small.in1k", "caformer_b36.sail_in22k_ft_in1k", "mobilenet-v2", "vgg16", "densenet169", "resnet-50"]
    # Treina todos modelos que podem ser usados no modelo multi-modal
    run_expirements(dataset_folder_path=dataset_folder_path, results_folder_path=results_folder_path, image_type=image_type, num_workers=num_workers, persistent_workers=True, num_epochs=num_epochs, type_of_problem=type_of_problem, batch_size=batch_size, k_folds=k_folds,
                    common_dim = common_dim, text_model_encoder=text_model_encoder, unfreeze_weights=unfreeze_weights, device=device, list_num_heads=list_num_heads, list_of_attention_mecanism=list_of_attention_mecanism, list_of_models=list_of_models)    
