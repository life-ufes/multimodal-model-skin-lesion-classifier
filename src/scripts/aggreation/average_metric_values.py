import os
import numpy as np
import pandas as pd

def get_dataset_content(dataset_path: str):
    # Read the CSV file
    dataset = pd.read_csv(dataset_path, sep=",")
    return dataset

def save_dataset(dataframe: pd.DataFrame, dataset_folder_path: str):
    # Print the new DataFrame
    print("\nFormatted Results DataFrame:")
    print(dataframe)
    
    # Optionally, save the results to a CSV file
    dataframe.to_csv(f"{dataset_folder_path}/formatted_results.csv", index=False)

if __name__ == "__main__":
    # Lista para armazenar todos os resultados
    all_results = []

    list_of_attention_mecanism = ["no-metadata", "concatenation", "weighted", "liwterm", "metablock", "crossattention", "gfcam","md-net", "metanet"]
    dataset_name = "PAD-UFES-20" # "PAD-UFES-20" # "PAD-UFES-25" # "ISIC-2019" # "MILK10K" # 
    num_heads = 8
    list_state_of_weights = ["unfrozen_weights"]
    # base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/{dataset_name}/multiclass/unfrozen_weights/{num_heads}"
    # base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/stratifiedkfold/2/all-weights-unfroozen/for_test/PAD-UFES-20/unfrozen_weights/{num_heads}"
    # base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/differents_dimensiond_of_projected_features/PAD-UFES-20/unfrozen_weights/8"
    # Path to your CSV file
    # base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/RG-ATT-512-EXPERIMENTS-07112025/{dataset_name}/unfrozen_weights/{num_heads}"
    # base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/MILK10k/dermoscopic/unfrozen_weights/{num_heads}"
    ## base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/01012026/{dataset_name}/unfrozen_weights/{num_heads}"
    ## base_folder_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/PAD-UFES-20/unfrozen_weights/8"
    # base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/different_features_with_dimension_size/PAD-UFES-20/unfrozen_weights/8"
    for status_of_weigths in list_state_of_weights:
        for common_size in [512]:
            for attention_mecanism in list_of_attention_mecanism:
                base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/artigo_1_GFCAM/10022026/{dataset_name}"
                # Testar com todos os modelos
                list_of_models = ["caformer_b36.sail_in22k_ft_in1k", "coat_lite_small.in1k", "davit_tiny.msft_in1k", "mvitv2_small.fb_in1k", "beitv2_large_patch16_224.in1k_ft_in22k_in1k", "efficientnet-b4", "densenet169", "mobilenet-v2", "resnet-50"]
                # list_of_models = [f"nas_multimodal_model_id-{int(i)}" for i in range(24)]
                # list_of_models = ["davit_tiny.msft_in1k"]
                for model_name in list_of_models:
                    # dataset_folder_path = f"{base_folder_path}/{attention_mecanism}/model_{model_name}_with_one-hot-encoder_{common_size}_with_best_architecture"
                    dataset_folder_path = f"{base_folder_path}/{status_of_weigths}/{num_heads}/{attention_mecanism}/model_{model_name}_with_one-hot-encoder_{common_size}_with_best_architecture"
                    dataset_path = os.path.join(dataset_folder_path, "model_metrics.csv")
                    # print(f"Dados do {model_name} e do mecanismo {attention_mecanism}")
                    try:
                        dataset = get_dataset_content(dataset_path=dataset_path)
                        
                        numeric_columns = [
                            "accuracy", "balanced_accuracy", "f1_score", "recall", "auc"
                        ]
                        
                        numeric_columns = [col for col in numeric_columns if col in dataset.columns]
                        
                        dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors='coerce')
                        
                        mean_values = dataset[numeric_columns].mean()
                        std_values = dataset[numeric_columns].std()
                        
                        # Format the result as "avg ± stv" for each metric
                        formatted_results = [f"{mean_values[col]:.4f} ± {std_values[col]:.4f}" for col in numeric_columns]
                        
                        # Cria o dataframe com os dados locais do modelo analisado
                        result_df = pd.DataFrame([formatted_results], columns=numeric_columns)
                        
                        # Adiciona o mecanismo de atenção e o nome do modelo para identificar os dados posteriormente
                        result_df['attention_mecanism'] = attention_mecanism
                        result_df['model_name'] = model_name
                        # result_df['common_size'] = common_size
                        result_df['state_of_the_weights'] = str(status_of_weigths)
                        
                        # Armazena os resultados na lista
                        all_results.append(result_df)
                    
                    except Exception as e:
                        print(f"Erro ao processar as métricas dos experimentos! Erro: {e}\n")
                        # Mesmo que dê erro, continua processando os resultados restantes
                        continue
        
# Concatenar todos os resultados em um único DataFrame
final_results_df = pd.concat(all_results, ignore_index=True)

# Reorganizar as colunas para colocar 'attention_mecanism' e 'model_name' como as primeiras
# final_results_df = final_results_df[['attention_mecanism', 'model_name', 'common_size'] + [col for col in final_results_df.columns if col not in ['attention_mecanism', 'model_name', 'common_size']]]

# Salvar os valores concatenados em um arquivo CSV
final_results_df.to_csv(f'{base_folder_path}/all_metric_values.csv', index=False)
print(f"Todos os resultados foram salvos em {base_folder_path}/all_metric_values.csv.")
