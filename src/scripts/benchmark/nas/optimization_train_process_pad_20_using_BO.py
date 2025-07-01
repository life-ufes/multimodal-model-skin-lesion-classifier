import torch
import torch.nn as nn
import os
import sys
# Adicione sys.path se necessário, mas geralmente é melhor configurar o PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import model_metrics, save_predictions # Assumindo que essas utilidades são flexíveis
from utils.early_stopping import EarlyStopping
from utils import load_local_variables
from models import dynamicMultimodalmodel # Importe o seu modelo dinâmico
from models import skinLesionDatasets, skinLesionDatasetsWithBert
from utils.save_model_and_metrics import save_model_and_metrics
import numpy as np
import random
import csv
import time
import json
from torch.utils.data import DataLoader
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real # Importe Real se for usar floats
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow
from tqdm import tqdm

# --- 0) Configurações Gerais e Semente ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
# --- Funções Auxiliares (mantidas como estão) ---
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

# --- 1) Função de Treinamento e Avaliação para Otimização Bayesiana ---

def train_and_evaluate_model(
    params: dict, # Parâmetros de arquitetura propostos pelo skopt
    fold_num:int,
    num_classes: int, train_loader:dict, val_loader:dict, device: str,
    targets:list, class_weights:list, results_folder_path:str,
    num_epochs_per_eval:int, num_metadata_features:int,
    model_name:str # Parâmetro fixo para esta rodada da BO
):
    # Desempacotar os parâmetros da lista (a ordem deve corresponder ao search_space_skopt)
    # Adapte os nomes e a ordem para o seu search_space_skopt
    # Exemplo:
    (num_blocks, initial_filters, kernel_size, num_heads, layers_per_block, 
     use_pooling, common_dim, attention_mecanism, 
     num_layers_text_fc, neurons_per_layer_size_of_text_fc, 
     num_layers_fc_module, neurons_per_layer_size_of_fc_module) = params # Ajuste a ordem aqui!

    config = {
        "num_blocks": str(num_blocks),
        "initial_filters": str(initial_filters),
        "kernel_size": str(kernel_size),
        "layers_per_block": str(layers_per_block),
        "use_pooling": str(use_pooling),
        "common_dim": str(common_dim),
        "attention_mecanism": str(attention_mecanism),
        "num_layers_text_fc": str(num_layers_text_fc),
        "neurons_per_layer_size_of_text_fc": str(neurons_per_layer_size_of_text_fc),
        "num_layers_fc_module": str(num_layers_fc_module),
        "neurons_per_layer_size_of_fc_module": str(neurons_per_layer_size_of_fc_module),
        # Adicione outros parâmetros fixos que seu modelo precise, mas não estão sendo otimizados
        "num_heads": str(num_heads), # Exemplo: se num_heads não está no search space
        "image_encoder": str(model_name) # Passa o encoder de imagem fixo
    }

    print(f"\n--- Avaliando Arquitetura Proposta ---")
    print(config) # Imprima a config para depuração

    # Instancia o modelo dinâmico com a configuração proposta
    # Adapte esta chamada ao seu dynamicMultimodalmodel.DynamicCNN
    model = dynamicMultimodalmodel.DynamicCNN(
        config=config, 
        num_classes=int(num_classes), 
        device=device,
        common_dim=int(common_dim), # Passado explicitamente ou via config
        num_heads=int(config["num_heads"]), # Usar o da config
        vocab_size=int(num_metadata_features), # Assumindo que isso é um valor fixo
        attention_mecanism=attention_mecanism, # Passado explicitamente ou via config
        n=1 if attention_mecanism == "no-metadata" else 2 # Exemplo da sua lógica
    )
    model.to(device)
    model_save_path = os.path.join(
        results_folder_path, 
        f"model_custom-cnn-with-NAS_with_one-hot-encoder_{common_dim}_with_best_architecture"
    )
    os.makedirs(model_save_path, exist_ok=True)
    print(model_save_path)


    # --- Treinamento do Modelo (Simplificado para a BO) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=False
    )
    
    # Early stopping (opcional, pode ser simplificado para o BO para apenas parar o treino)
    # Se o EarlyStopping salvar em disco, isso pode ficar muito lento e gerar muitos arquivos.
    # É melhor ter um EarlyStopping que apenas retorne um sinal para parar o loop.
    # Ou desabilitar save_to_disk para o contexto da BO.
    # Para o BO, o objetivo é uma avaliação RÁPIDA do desempenho, não o melhor modelo salvo.
    early_stopping = EarlyStopping(
        patience=5, # Reduzido para BO para acelerar
        delta=0.001,
        verbose=False,
        path=None,
        save_to_disk=False,
        early_stopping_metric_name="val_bacc"
    )

    current_best_val_bacc = 0.0 # Usaremos para o early stopping simplificado se não usar a classe

    initial_time = time.time()

    for epoch_index in range(num_epochs_per_eval): # Usar TRAIN_EPOCHS_PER_EVAL aqui
        model.train()
        running_loss = 0.0
        for _, image, metadata, label in train_loader:
            image, metadata, label = image.to(device), metadata.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(image, metadata)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, image, metadata, label in val_loader:
                image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

        # Avaliar métricas para Early Stopping
        metrics, _, _ = model_metrics.evaluate_model(
            model=model, dataloader=val_loader, device=device,
            fold_num=fold_num, # fold_num fictício para BO
            targets=targets,
            base_dir=model_save_path, # Não salva dados intermediários
            model_name=model_name
        )
        current_val_bacc = float(metrics["balanced_accuracy"])

        # Usar o EarlyStopping para decidir se para o treino desta arquitetura
        early_stopping(val_loss=val_loss, val_bacc=current_val_bacc, model=model)
        if early_stopping.early_stop:
            print(f"    Early stopping para esta arquitetura na época {epoch_index+1}")
            break
    
    train_process_time = time.time() - initial_time

    # Carrega o melhor modelo encontrado
    model = early_stopping.load_best_weights(model)

    # Avaliação FINAL do modelo para a BO
    model.eval()
    metrics, all_labels, all_predictions = model_metrics.evaluate_model(
        model=model, dataloader=val_loader, device=device,
        fold_num=fold_num, targets=targets,
        base_dir=model_save_path, model_name=model_name
    )
    final_balanced_accuracy = float(metrics["balanced_accuracy"])
    
    print(f"  Acurácia Balanceada Final: {final_balanced_accuracy:.4f}")
    
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

    # Limpar a memória da GPU para a próxima avaliação
    del model
    torch.cuda.empty_cache()

    # Salvar as métricas
    metrics_file = os.path.join(results_folder_path, "all_model_metrics.csv")
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
        
    with open(os.path.join(folder_path, "config.json"), "w") as f:
        json.dump(dict(config), f, indent=2)

    # Contabiliza um novo fold
    global global_fold_counter
    fold_num = global_fold_counter
    global_fold_counter += 1
    # Retorna 1.0 - acurácia para minimização (o gp_minimize buscará o menor valor)
    return 1.0 - final_balanced_accuracy

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- Carrega Variáveis Locais e Configurações Fixas ---
    local_variables = load_local_variables.get_env_variables()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    # Parâmetros fixos para esta execução da otimização Bayesiana
    NUM_EPOCHS_PER_EVAL = 5 # Número de épocas para treinar CADA arquitetura candidata
    global_fold_counter = 0
    NUMBER_OF_TRIES = 5 # Número de arquiteturas a serem avaliadas pela BO. Aumente para resultados melhores.
    print(f"\nIniciando Otimização Bayesiana com {NUMBER_OF_TRIES} avaliações...")
    n_initial_points = 5
    BATCH_SIZE = local_variables["batch_size"]
    # k_folds = 1 # Para a BO, geralmente não usamos k-folds para a avaliação interna
    TEXT_MODEL_ENCODER = 'one-hot-encoder'
    UNFREEZE_WEIGHTS = bool(local_variables["unfreeze_weights"])
    LLM_MODEL_NAME_SEQUENCE_GENERATOR = local_variables["llm_model_name_sequence_generator"]
    RESULTS_FOLDER_PATH = local_variables["results_folder_path"]
    RESULTS_FOLDER_PATH = f"{RESULTS_FOLDER_PATH}/{local_variables['dataset_folder_name']}/{'unfrozen_weights' if UNFREEZE_WEIGHTS else 'frozen_weights'}"
    NUM_WORKERS = 1
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = local_variables["dataset_folder_path"]
    list_num_heads = local_variables["list_num_heads"]

    # --- Preparação do Dataset (Feito uma única vez, fora da função objetivo) ---
    # Adapte a criação do dataset conforme o seu código original
    if (TEXT_MODEL_ENCODER in ['one-hot-encoder', "tab-transformer"]):
        dataset = skinLesionDatasets.SkinLesionDataset(
            metadata_file=f"{dataset_folder_path}/metadata.csv",
            img_dir=f"{dataset_folder_path}/images",
            bert_model_name=TEXT_MODEL_ENCODER,
            image_encoder="placeholder_for_BO", # Coloque um placeholder ou um valor padrão se seu dataset precisar
            drop_nan=False,
            size=(224,224))
    elif (TEXT_MODEL_ENCODER in ['gpt2', 'bert-base-uncased']):
        dataset = skinLesionDatasetsWithBert.SkinLesionDataset(
            metadata_file=f"{dataset_folder_path}/metadata_with_sentences_new-prompt-{LLM_MODEL_NAME_SEQUENCE_GENERATOR}.csv",
            img_dir=f"{dataset_folder_path}/images",
            bert_model_name=TEXT_MODEL_ENCODER,
            image_encoder="placeholder_for_BO",
            drop_nan=False,
            size=(224,224))
    else:
        raise ValueError("Encoder de texto não implementado!\n")
    
    num_metadata_features = dataset.features.shape[1] if TEXT_MODEL_ENCODER == 'one-hot-encoder' else 512
    num_classes = len(dataset.metadata['diagnostic'].unique())

    labels_full_dataset = [dataset.labels[i] for i in range(len(dataset))]
    
    # Split simples (train/val) para a BO
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2, # Proporção da validação para a busca BO
        stratify=labels_full_dataset,
        random_state=42
    )
    
    train_dataset = type(dataset)(
        metadata_file=dataset.metadata_file, img_dir=dataset.img_dir, size=(224,224),
        drop_nan=dataset.is_to_drop_nan, bert_model_name=dataset.bert_model_name,
        image_encoder=dataset.image_encoder, is_train=True
    )
    train_dataset.metadata = dataset.metadata.iloc[train_idx].reset_index(drop=True)
    train_dataset.features, train_dataset.labels, train_dataset.targets = train_dataset.one_hot_encoding()

    val_dataset = type(dataset)(
        metadata_file=dataset.metadata_file, img_dir=dataset.img_dir, size=(224,224),
        drop_nan=dataset.is_to_drop_nan, bert_model_name=dataset.bert_model_name,
        image_encoder=dataset.image_encoder, is_train=False
    )
    val_dataset.metadata = dataset.metadata.iloc[val_idx].reset_index(drop=True)
    val_dataset.features, val_dataset.labels, val_dataset.targets = val_dataset.one_hot_encoding()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)

    train_labels_for_weights = [labels_full_dataset[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels_for_weights, num_classes).to(device)
    print(f"Pesos das classes: {class_weights}")
    
    # --- Definindo o Espaço de Busca para a Otimização Bayesiana ---
    # Aqui, a ordem dos parâmetros nesta lista DEVE CORRESPONDER
    # à ordem em que você desempacota 'params' na 'train_and_evaluate_model'.
    search_space_skopt = [
        Categorical(categories=[2, 3, 5],name="num_blocks"), # Aumentei o range
        Categorical(categories=[16, 32, 64], name="initial_filters"), # Use Integer para valores discretos
        Categorical(categories=[3, 5], name="kernel_size"),
        Categorical(categories=list_num_heads, name="num_heads"),
        Integer(low=1, high=3, name="layers_per_block"), # Aumentei o range
        Categorical(categories=[True, False], name="use_pooling"),
        Categorical(categories=[64, 128, 256, 512], name="common_dim"), # common_dim agora é otimizado
        Categorical(categories=["concatenation", "crossattention", "metablock", "gfcam", "no-metadata", "weighted"], name="attention_mecanism"), # Incluí mais opções
        Integer(low=1, high=3, name="num_layers_text_fc"), # Exemplo: 1 a 3 camadas FC para texto
        Categorical(categories=[128, 256, 512], name="neurons_per_layer_size_of_text_fc"), # Neurônios FC para texto
        Integer(low=1, high=3, name="num_layers_fc_module"), # Exemplo: 1 a 3 camadas FC para a cabeça final
        Categorical(categories=[256, 512], name="neurons_per_layer_size_of_fc_module") # Neurônios FC para a cabeça final
    ]

    # --- Executando a Otimização Bayesiana ---
    # Usamos uma função lambda para passar os argumentos fixos para train_and_evaluate_model
    # enquanto gp_minimize passa apenas os 'params' que ele está otimizando.
    res = gp_minimize(
        func=lambda params: train_and_evaluate_model(
            params=params,
            fold_num=global_fold_counter,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            targets=dataset.targets, # Passa os targets do dataset
            class_weights=class_weights,
            results_folder_path=RESULTS_FOLDER_PATH,
            num_epochs_per_eval=NUM_EPOCHS_PER_EVAL,
            num_metadata_features=num_metadata_features,
            model_name="custom-cnn-with-NAS" # Passa o nome do image_encoder fixo
        ),
        dimensions=search_space_skopt,
        n_calls=NUMBER_OF_TRIES,
        n_initial_points=n_initial_points, # Número de pontos iniciais aleatórios
        acq_func="gp_hedge", # Estratégia de aquisição
        random_state=42
    )

    # --- Exibindo os Resultados Finais da Otimização Bayesiana ---
    print("\n" + "="*60)
    print("             RESULTADOS FINAIS DA OTIMIZAÇÃO BAYESIANA          ")
    print("="*60)
    # Converta o valor minimizado de volta para a acurácia
    best_balanced_accuracy = 1.0 - res.fun
    print(f"Melhor Acurácia Balanceada de Validação encontrada: {best_balanced_accuracy:.4f}")
    print("\nMelhores Hiperparâmetros da Arquitetura:")
    best_params_dict = dict(zip([dim.name for dim in search_space_skopt], res.x))
    for param_name, param_value in best_params_dict.items():
        print(f"  {param_name}: {param_value}")

    # Salvar os melhores parâmetros e a melhor acurácia
    best_result_path = os.path.join(RESULTS_FOLDER_PATH, "best_nas_result.json")
    with open(best_result_path, "w") as f:
        json.dump({"best_balanced_accuracy": best_balanced_accuracy, "best_params": best_params_dict}, f, indent=2, cls=NumpyEncoder)
    print(f"\nMelhores resultados salvos em: {best_result_path}")

    # --- Plotando a Curva de Convergência ---
    plt.figure(figsize=(10, 5))
    # Converter os valores da função objetivo (1 - acurácia) de volta para acurácia
    plt.plot(np.arange(1, len(res.func_vals)+1), 1.0 - np.array(res.func_vals), marker='o', linestyle='-')
    plt.title("Convergência da Otimização Bayesiana (Melhor Acurácia Balanceada)")
    plt.xlabel("Número de Avaliações (Arquiteturas Treinadas)")
    plt.ylabel("Melhor Acurácia Balanceada Encontrada até o Momento")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_FOLDER_PATH, "bo_convergence_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfico de convergência salvo em: {plot_path}")
    plt.show()

    # # MLflow logging para a run principal da BO
    # with mlflow.start_run(run_name="Bayesian_Optimization_NAS_Run"):
    #     mlflow.log_param("total_evaluations", NUMBER_OF_TRIES)
    #     mlflow.log_metric("best_balanced_accuracy", best_balanced_accuracy)
    #     mlflow.log_params(best_params_dict)
    #     mlflow.log_artifact(plot_path)
    #     mlflow.log_artifact(best_result_path)
    #     # Log do espaço de busca
    #     mlflow.log_param("search_space_definition", json.dumps(
    #         {dim.name: (dim.low, dim.high) if isinstance(dim, (Integer, Real)) else dim.categories for dim in search_space_skopt}
    #     ))

    print("\nOtimização Bayesiana concluída. Verifique os logs do MLflow para mais detalhes.")
