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

    list_of_attention_mecanism = ["att-intramodal+residual", "att-intramodal+residual+cross-attention-metadados", "att-intramodal+residual+cross-attention-metadados+att-intramodal+residual", "no-metadata", "cross-weights-after-crossattention", "concatenation", "weighted", "weighted-after-crossattention", "crossattention"]
    dataset_name = "ISIC-2019" # "PAD-UFES-20"# "ISIC-2020" # "PAD-UFES-25" # "ISIC-2019" # "PAD-UFES-20"
    num_heads = 8
    # base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/{dataset_name}/multiclass/unfrozen_weights/{num_heads}"
    # base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/stratifiedkfold/2/all-weights-unfroozen/for_test/PAD-UFES-20/unfrozen_weights/{num_heads}"
    # base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/differents_dimensiond_of_projected_features/PAD-UFES-20/unfrozen_weights/8"
    # Path to your CSV file
    base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/differents_dimension_of_projected_features/{dataset_name}/unfrozen_weights/{num_heads}"
    # base_folder_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/teste_com_val_bacc/PAD-UFES-20/unfrozen_weights/8"
    for common_size in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        for attention_mecanism in list_of_attention_mecanism:
            # Testar com todos os modelos
            list_of_models = ["mvitv2_small.fb_in1k", "coat_lite_small.in1k","davit_tiny.msft_in1k", "caformer_b36.sail_in22k_ft_in1k", "beitv2_large_patch16_224.in1k_ft_in22k_in1k", "vgg16", "mobilenet-v2", "densenet169", "resnet-50"]
            
            for model_name in list_of_models:
                dataset_folder_path = f"{base_folder_path}/{attention_mecanism}/model_{model_name}_with_one-hot-encoder_{common_size}_with_best_architecture"
                dataset_path = os.path.join(dataset_folder_path, "model_metrics.csv")
                print(f"Dados do {model_name} e do mecanismo {attention_mecanism}")
                try:
                    dataset = get_dataset_content(dataset_path=dataset_path)
                    
                    numeric_columns = [
                        "accuracy", "balanced_accuracy", "f1_score", "precision", "recall", "auc",
                        "train_loss", "val_loss", "train process time", "epochs"
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
                    result_df['common_size'] = common_size

                    
                    # Armazena os resultados na lista
                    all_results.append(result_df)
                
                except Exception as e:
                    print(f"Erro ao processar as métricas dos experimentos! Erro: {e}\n")
                    # Mesmo que dê erro, continua processando os resultados restantes
                    continue
        
    # Concatenar todos os resultados em um único DataFrame
    final_results_df = pd.concat(all_results, ignore_index=True)
    
    # Reorganizar as colunas para colocar 'attention_mecanism' e 'model_name' como as primeiras
    final_results_df = final_results_df[['attention_mecanism', 'model_name', 'common_size'] + [col for col in final_results_df.columns if col not in ['attention_mecanism', 'model_name', 'common_size']]]
    
    # Salvar os valores concatenados em um arquivo CSV
    final_results_df.to_csv(f'{base_folder_path}/all_metric_values.csv', index=False)
    print(f"Todos os resultados foram salvos em {base_folder_path}/all_metric_values.csv.")
