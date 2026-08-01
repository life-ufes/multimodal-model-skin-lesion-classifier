import torch
import torch.nn as nn
from utils import model_metrics
from utils.early_stopping import EarlyStopping
from utils import load_local_variables
from models import multimodalIntraModalWithPubMedBert
from models import skinLesionDatasetsWithSentenceEmbeddings
from models.loadImageModelClassifier import PUBMEDBERT_ENCODERS
from utils.save_model_and_metrics import save_model_and_metrics
from sklearn.model_selection import StratifiedGroupKFold
import time
import os
import traceback
from torch.utils.data import DataLoader, Subset
import numpy as np
import mlflow
from tqdm import tqdm

# Encoders de texto suportados por este script (embeddings pré-computados).
SENTENCE_EMBEDDING_ENCODERS = list(PUBMEDBERT_ENCODERS)

# Mecanismos de fusão implementados por multimodalIntraModalWithPubMedBert.
# Importado do próprio modelo para que a lista nunca fique atrás do catálogo.
SUPPORTED_ATTENTION_MECANISMS = set(multimodalIntraModalWithPubMedBert.ATTENTION_MECANISMS)

# Modos válidos de fine-tuning do backbone (ver loadModels.set_backbone_train_mode)
VALID_BACKBONE_TRAIN_MODES = {
    "frozen_weights",
    "unfrozen_weights",
    "last_layer_unfrozen_weights",
    "partial",
}


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
                  results_folder_path,
                  dataset_folder_name):

    criterion = nn.CrossEntropyLoss(weight=weightes_per_category)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    # CORREÇÃO (Parte 8a): `verbose` foi depreciado no PyTorch 2.2 e removido
    # em versões posteriores; o print do LR abaixo já cumpre a mesma função.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2
    )
    model.to(device)

    model_save_path = os.path.join(
        results_folder_path,
        f"model_{model_name}_with_{text_model_encoder}_{common_dim}_with_best_architecture"
    )
    os.makedirs(model_save_path, exist_ok=True)
    print(model_save_path)

    early_stopping = EarlyStopping(
        patience=5,
        delta=0.01,
        verbose=True,
        path=str(model_save_path + f'/{model_name}_fold_{fold_num}/best-model/'),
        save_to_disk=True,
        early_stopping_metric_name="val_loss"
    )

    initial_time = time.time()
    epoch_index = 0
    train_losses, val_losses = [], []

    # CORREÇÃO (Parte 8b): `dataset_folder_name` era lido como global definido
    # apenas sob `__main__`, quebrando o módulo se importado.
    experiment_name = f"EXPERIMENTOS-{dataset_folder_name}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=(
            f"image_extractor_model_{model_name}_with_mecanism_"
            f"{attention_mecanism}_fold_{fold_num}_num_heads_{num_heads}"
        )
    ):
        mlflow.log_param("fold_num", fold_num)
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

            for batch_index, (_, image, metadata, label) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch_index+1}/{num_epochs}", leave=False)):
                image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(float(train_loss))
            print(f"\nTraining: Epoch {epoch_index}, Loss: {train_loss:.4f}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _, image, metadata, label in val_loader:
                    image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            val_losses.append(float(val_loss))
            print(f"Validation Loss: {val_loss:.4f}")

            scheduler.step(val_loss)
            current_lr = [pg['lr'] for pg in optimizer.param_groups]
            print(f"Current Learning Rate(s): {current_lr}\n")

            metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
                model=model, dataloader=val_loader, device=device, fold_num=fold_num,
                targets=targets, base_dir=model_save_path, model_name=model_name
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
        metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
            model=model, dataloader=val_loader, device=device, fold_num=fold_num,
            targets=targets, base_dir=model_save_path, model_name=model_name
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
        all_probabilities=all_probs,
        targets=targets,
        data_val="val",
        train_losses=train_losses,
        val_losses=val_losses
    )
    print(f"Model saved at {model_save_path}")

    return model, model_save_path


def pipeline(dataset_train, dataset_eval, num_metadata_features, num_epochs, batch_size, device,
             k_folds, num_classes, model_name, num_heads, common_dim, text_model_encoder,
             unfreeze_weights, attention_mecanism, results_folder_path, dataset_folder_name,
             num_workers=10, persistent_workers=False):
    # Rótulos, grupos e alvos vêm sempre da instância de avaliação: as duas
    # instâncias compartilham o mesmo CSV e diferem apenas nas transforms.
    labels = [dataset_eval.labels[i] for i in range(len(dataset_eval))]
    groups = dataset_eval.metadata["patient_id"].values  # agrupa por paciente
    stratifiedKFold = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(stratifiedKFold.split(range(len(dataset_eval)), labels, groups=groups)):
        print(f"Fold {fold+1}/{k_folds}")
        # CORREÇÃO (Parte 5): o subset de treino usa a instância com
        # `is_train=True` (com augmentation); a validação usa a determinística.
        train_subset = Subset(dataset_train, train_idx)
        val_subset = Subset(dataset_eval, val_idx)

        # CORREÇÃO (Parte 6): `drop_last=True` no treino evita batch de tamanho
        # 1, que faz o BatchNorm1d de fc_fusion levantar ValueError.
        # `persistent_workers` já estava na assinatura mas não era repassado.
        use_persistent = bool(persistent_workers) and num_workers > 0
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
            persistent_workers=use_persistent, pin_memory=True)
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=True,
            persistent_workers=use_persistent, pin_memory=True)

        train_labels = [labels[i] for i in train_idx]
        class_weights = compute_class_weights(train_labels, num_classes).to(device)
        print(f"Pesos das classes no fold {fold+1}: {class_weights}")

        if text_model_encoder not in SENTENCE_EMBEDDING_ENCODERS:
            raise ValueError(
                f"Encoder de texto não implementado: {text_model_encoder!r}. "
                f"Esperado um de {sorted(SENTENCE_EMBEDDING_ENCODERS)}"
            )

        # CORREÇÃO: `multimodalIntraModalWithBert` tokeniza — seu forward acessa
        # `metadata['input_ids']`, enquanto este dataset entrega um tensor float
        # (B, D) já embutido. Quem trata o caso pré-computado é
        # `multimodalIntraModalWithPubMedBert` (use_precomputed_text_embedding).
        model = multimodalIntraModalWithPubMedBert.MultimodalModel(
            num_classes, num_heads, device,
            cnn_model_name=model_name,
            text_model_name=text_model_encoder,
            common_dim=common_dim,
            # `vocab_size` carrega a dimensão real do embedding do dataset:
            # é ela que dimensiona o ramo textual quando o encoder é None.
            vocab_size=num_metadata_features,
            unfreeze_weights=unfreeze_weights,
            attention_mecanism=attention_mecanism,
            # A cabeça de "no-metadata" consome um único vetor; o próprio modelo
            # trata esse caso a partir de MECHANISM_SPECS, então n=2 aqui vale
            # para todos os mecanismos que concatenam imagem+texto.
            n=2
        )

        # Treino do modelo carregado
        model, _ = train_process(
            num_epochs=num_epochs, num_heads=num_heads, fold_num=fold+1,
            train_loader=train_loader, val_loader=val_loader,
            targets=dataset_eval.targets, model=model, device=device,
            weightes_per_category=class_weights, common_dim=common_dim,
            model_name=model_name, text_model_encoder=text_model_encoder,
            attention_mecanism=attention_mecanism,
            results_folder_path=results_folder_path,
            dataset_folder_name=dataset_folder_name
        )


def run_expirements(dataset_folder_path: str, results_folder_path: str,
                    llm_model_name_sequence_generator: str, num_epochs: int, batch_size: int,
                    k_folds: int, common_dim: int, text_model_encoder: str, unfreeze_weights: str,
                    device, list_num_heads: list, list_of_attention_mecanism: list,
                    list_of_models: list, dataset_folder_name: str):
    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    if text_model_encoder not in SENTENCE_EMBEDDING_ENCODERS:
                        raise ValueError(
                            f"Encoder de texto não implementado: {text_model_encoder!r}. "
                            f"Esperado um de {sorted(SENTENCE_EMBEDDING_ENCODERS)}"
                        )
                    if attention_mecanism not in SUPPORTED_ATTENTION_MECANISMS:
                        raise ValueError(
                            f"Mecanismo {attention_mecanism!r} não implementado no ramo de "
                            f"sentence embeddings. Esperado um de "
                            f"{sorted(SUPPORTED_ATTENTION_MECANISMS)}"
                        )

                    # CORREÇÃO (Parte 5): duas instâncias do dataset — treino com
                    # augmentation, avaliação determinística. `_cache_key` não
                    # depende de `is_train`, então a segunda instância reaproveita
                    # o cache de embeddings em disco (sem custo extra de encoding).
                    common_kwargs = dict(
                        metadata_file=f"{dataset_folder_path}/metadata_with_sentences.csv",
                        img_dir=f"{dataset_folder_path}/images",
                        bert_model_name=text_model_encoder,
                        image_encoder=model_name,
                        drop_nan=False,
                    )
                    dataset_train = skinLesionDatasetsWithSentenceEmbeddings.SkinLesionDataset(
                        **common_kwargs, is_train=True)
                    dataset_eval = skinLesionDatasetsWithSentenceEmbeddings.SkinLesionDataset(
                        **common_kwargs, is_train=False)

                    # A dimensão do metadado textual vem do próprio dataset, que
                    # a lê do SentenceTransformer. Um fallback fixo (512) só
                    # mascararia o erro: o ramo textual seria dimensionado
                    # errado e a falha apareceria como mismatch de shape no
                    # primeiro batch, longe da causa.
                    num_metadata_features = int(dataset_eval.embedding_dim)
                    print(f"Número de features do metadados: {num_metadata_features}\n")
                    num_classes = len(dataset_eval.metadata['diagnostic'].unique())

                    pipeline(
                        dataset_train=dataset_train,
                        dataset_eval=dataset_eval,
                        num_metadata_features=num_metadata_features,
                        num_epochs=num_epochs, batch_size=batch_size,
                        device=device, k_folds=k_folds, num_classes=num_classes,
                        model_name=model_name, common_dim=common_dim,
                        text_model_encoder=text_model_encoder,
                        num_heads=num_heads,
                        unfreeze_weights=unfreeze_weights,
                        attention_mecanism=attention_mecanism,
                        results_folder_path=f"{results_folder_path}/{num_heads}/{attention_mecanism}",
                        dataset_folder_name=dataset_folder_name
                    )
                # CORREÇÃO (Parte 7): sem o traceback, uma falha de dimensão no
                # ramo textual era indistinguível de uma imagem corrompida.
                except Exception:
                    print(f"Erro no modelo {model_name} / mecanismo {attention_mecanism} "
                          f"/ heads {num_heads}:")
                    traceback.print_exc()
                    continue


if __name__ == "__main__":
    # Carrega os dados localmente
    local_variables = load_local_variables.get_env_variables()
    num_epochs = local_variables["num_epochs"]
    batch_size = local_variables["batch_size"]
    k_folds = local_variables["k_folds"]
    common_dim = local_variables["common_dim"]
    list_num_heads = local_variables["list_num_heads"]
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = local_variables["dataset_folder_path"]
    unfreeze_weights = str(local_variables["unfreeze_weights"])
    llm_model_name_sequence_generator = local_variables["LLM_MODEL_NAME_SEQUENCE_GENERATOR"]

    if unfreeze_weights not in VALID_BACKBONE_TRAIN_MODES:
        raise ValueError(
            f"unfreeze_weights inválido: {unfreeze_weights!r}. "
            f"Esperado um de {sorted(VALID_BACKBONE_TRAIN_MODES)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for text_model_encoder in ['pubmedbert-base-embeddings-100K']:

        results_folder_path = (
            f"./data/results/generated-senteces/{dataset_folder_name}"
            f"/textual-encoder-{text_model_encoder}/{unfreeze_weights}"
        )

        # Para todos os tipos de estratégias a serem usadas.
        # "rg-dermnet" é o alias do modelo proposto (RG-ATT completo:
        # self-att -> RG-residual -> cross-att) — ver MECHANISM_ALIASES em
        # multimodalIntraModalWithPubMedBert.
        list_of_attention_mecanism = ["rg-att-cross-modal"]
        # Testar com todos os modelos
        list_of_models = ["davit_tiny.msft_in1k"]

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
            dataset_folder_name=dataset_folder_name
        )