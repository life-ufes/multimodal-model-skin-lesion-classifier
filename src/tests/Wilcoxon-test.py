import numpy as np
from stats import statistical_test

# Define as métricas (média e desvio padrão) para cada modelo
metrics = {
    "Our Model": {"ACC": (0.8194, 0.0169), "BACC": (0.8117, 0.0179), "AUC": (0.9494, 0.0057)},
    "No Metadata": {"ACC": (0.6160, 0.0510), "BACC": (0.6510, 0.0500), "AUC": (0.901, 0.0070)},
    "Concatenation": {"ACC": (0.7410, 0.0140), "BACC": (0.7280, 0.0290), "AUC": (0.9290, 0.0060)},
    "MetaBlock": {"ACC": (0.7350, 0.0130), "BACC": (0.7650, 0.0170), "AUC": (0.9350, 0.0040)},
    "MetaNet": {"ACC": (0.7320, 0.0540), "BACC": (0.7420, 0.0190), "AUC": (0.9360, 0.0060)},
    "Fully-CrossAttention": {"ACC": (0.7680, 0.0220), "BACC": (0.7750, 0.0220), "AUC": (0.9470, 0.0070)},
}

def collect_samples(model_data):
    """
    Para cada métrica (ACC, BACC, AUC), coleta os dois valores (média e desvio padrão)
    e os armazena em uma lista. Assim, cada modelo terá 6 amostras.
    """
    samples = []
    for metric in ["ACC", "BACC", "AUC"]:
        mean_val, std_val = model_data[metric]
        samples.append(mean_val)
        samples.append(std_val)
    return samples

def main():
    # Constrói a matriz de dados: cada linha representa um modelo e cada coluna uma "amostra"
    all_samples = []
    alg_names = []
    for model, values in metrics.items():
        samples = collect_samples(values)
        all_samples.append(samples)
        alg_names.append(model)

    data_array = np.array(all_samples)

    print("Data array (transposed):")
    print(data_array.transpose())
    print("\nAlgorithm names:")
    print(alg_names)
    # Realiza o teste estatístico entre todos os modelos
    out = statistical_test(data=data_array.transpose(), alg_names=alg_names, pv_wilcoxon=0.1)
if __name__=="__main__":
    main()