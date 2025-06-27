import os
import stats
import numpy as np
import pandas as pd

def load_dataset(file_path):
    """Loads dataset from the given file path"""
    try:
        dataset = pd.read_csv(file_path, sep=",")
        return dataset
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def get_metric_values(file_folder_path, file_name='model_metrics.csv', metric_names=['acc', 'bacc', 'auc']):
    """
    Retrieves specified metric values from the 'model_metrics.csv' file in the given folder.
    Handles multiple metric names.
    """
    list_of_values_wanted_metrics = []  # A list to store values for all metrics
    csv_file = os.path.join(file_folder_path, file_name)
    
    if os.path.exists(csv_file):  # Check if the CSV file exists
        data = load_dataset(csv_file)
        
        if data is not None:  # Ensure data is successfully loaded
            for metric_name in metric_names:
                # Check if the requested metric exists in the columns
                if metric_name in data.columns:
                    # Exclude the last row and append the values to the list
                    list_of_values_wanted_metrics.append(data[metric_name].iloc[:].values)
                else:
                    print(f"{metric_name} not found in {csv_file}")
        else:
            print("Failed to load data from the CSV file.")
    else:
        print(f"CSV file not found: {csv_file}")
    
    # Return the list of metrics' values flattened
    return np.array(list_of_values_wanted_metrics).ravel()

def load_models_metrics(file_folder_path, wanted_metric_list):
    list_of_values_wanted_metric = []
    if os.path.exists(file_folder_path):
        csv_file = os.path.join(file_folder_path, 'cv-results.csv')
        if os.path.exists(csv_file):  # Verifica se o arquivo CSV existe
            data = load_dataset(csv_file)
            for metric_name in wanted_metric_list:
                if metric_name in data.columns:
                    # Exclui as duas últimas linhas
                    aux = data[metric_name].iloc[:-2].values
                    list_of_values_wanted_metric.append(aux)
                else:
                    print(f"{metric_name} não encontrada no arquivo {csv_file}\n")
        else:
            print(f"Arquivo CSV não encontrado: {csv_file}\n")
    else:
        print(f"Pasta não encontrada: {file_folder_path}\n")
        
    return np.array(list_of_values_wanted_metric).ravel()


def save_statistics_tests(test_results:str, file_folder_path: str):
    '''
        Função para salvar os resultados dos testes estatísticos
    '''
    try:
        dataframe = pd.DataFrame(data=test_results)
        dataframe.to_csv(file_folder_path+f"_statistics_tests_results.csv")
    except Exception as e:
        print(f"Erro salvar os dados. Erro: {e}\n")

if __name__ == "__main__":
    # Onde os dados do teste serão salvos
    base_file_folder_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/tests/results"
    wanted_metric_list = ['accuracy','balanced_accuracy','f1_score','recall','auc']
    
    
    file_folder_path="/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/PAD-UFES-20/unfrozen_weights/2/att-intramodal+residual+cross-attention-metadados/model_davit_tiny.msft_in1k_with_one-hot-encoder_512_with_best_architecture"
    # Variáveis a serem selecionadas
    # Load dataset (though not used directly here)
    file_content = load_dataset(file_folder_path+"/model_metrics.csv")
    # Carregar os dados do modelo multimodal - PAD-UFES-20
    # Get metric values for the specified metrics
    aux_metric_values_multimodal_model = get_metric_values(file_folder_path, metric_names=wanted_metric_list)
    list_of_used_algs = []
    list_all_models_metrics_all_lists=[]
    list_of_used_algs.append("our-method")
    list_all_models_metrics_all_lists.append(aux_metric_values_multimodal_model)
    
    # Path to the folder containing the results
    file_folder_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/unfrozen-weights/2/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture"
    # Variáveis a serem selecionadas
    # Load dataset (though not used directly here)
    file_content = load_dataset(file_folder_path+"/model_metrics.csv")
    # Carregar os dados do modelo multimodal - PAD-UFES-20
    # Get metric values for the specified metrics
    aux_metric_values_multimodal_model = get_metric_values(file_folder_path, metric_names=wanted_metric_list)
    # list_of_used_algs = []
    # list_all_models_metrics_all_lists=[]
    list_of_used_algs.append("gated-cross-attention")
    list_all_models_metrics_all_lists.append(aux_metric_values_multimodal_model)
    
    # MD-Net results
    file_folder_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/MD-Net/PAD-UFES-20/frozen_weights/0/md-net/model_densenet169_with_one-hot-encoder_512_with_best_architecture"
    # Variáveis a serem selecionadas
    # Load dataset (though not used directly here)
    file_content = load_dataset(file_folder_path+"/model_metrics.csv")
    # Carregar os dados do modelo multimodal - PAD-UFES-20
    # Get metric values for the specified metrics
    aux_metric_values_multimodal_model = get_metric_values(file_folder_path, metric_names=wanted_metric_list)
    list_of_used_algs.append("md-net")
    list_all_models_metrics_all_lists.append(aux_metric_values_multimodal_model)

    # ## Add dos dados do do trabalho 'A deep learning based multimodal ....'
    file_folder_path="/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/a-deep-learning-based-multimodal/PAD-UFES-20/frozen_weights/2/concatenation/model_resnet-50_with_one-hot-encoder_512_with_best_architecture"
    # Load dataset (though not used directly here)
    file_content = load_dataset(file_folder_path+"/model_metrics.csv")
    # Carregar os dados do modelo multimodal - PAD-UFES-20
    # Get metric values for the specified metrics
    aux_metric_values_multimodal_model = get_metric_values(file_folder_path, metric_names=wanted_metric_list)
    list_of_used_algs.append("a-deep-learning-based-multimodal")
    list_all_models_metrics_all_lists.append(aux_metric_values_multimodal_model)

    file_folder_path="/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/tests/results/PAD-20/lighterm"
    # Load dataset (though not used directly here)
    file_content = load_dataset(file_folder_path+"/model_metrics.csv")
    # Carregar os dados do modelo multimodal - PAD-UFES-20
    # Get metric values for the specified metrics
    aux_metric_values_multimodal_model = get_metric_values(file_folder_path, metric_names=wanted_metric_list)
    list_of_used_algs.append("lightwer")
    list_all_models_metrics_all_lists.append(aux_metric_values_multimodal_model)


    for alg in ["no-metadata", "concat", "metanet", "metablock"]:
        file_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/tests/results/PAD-20/{alg}"
        # Concat, Metablock, MetaNet
        list_of_used_algs.append(alg)
        wanted_metric_list = ['acc', 'bacc', 'weighted avg f1-score', 'weighted avg recall', 'auc']
        all_models_metrics=load_models_metrics(file_folder_path=file_folder_path, wanted_metric_list=wanted_metric_list)
        list_all_models_metrics_all_lists.append(all_models_metrics)


    out=stats.statistical_test(data=list_all_models_metrics_all_lists, alg_names=list_of_used_algs)
    # Salvar os resultados
    save_statistics_tests(test_results={out}, file_folder_path=base_file_folder_path)

    