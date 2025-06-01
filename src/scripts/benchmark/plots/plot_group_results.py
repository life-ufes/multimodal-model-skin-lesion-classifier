import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Caminho para o arquivo CSV
csv_file_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/differents_dimensiond_of_projected_features/PAD-UFES-20/unfrozen_weights/8/all_metric_values.csv"

# Carregar CSV
df = pd.read_csv(csv_file_path)

# Extrair média do 'balanced_accuracy' (remove ± e pega o primeiro número)
df['BACC_mean'] = df['balanced_accuracy'].apply(lambda x: float(re.match(r'([0-9.]+)', x).group(1)))

# Criar um rótulo combinado para diferenciar os modelos com mecanismos de atenção
df['Model+Attention'] = df['model_name'] + "\n(" + df['attention_mecanism'] + ")"

# Ordenar para manter consistência visual
df = df.sort_values(by=['common_size', 'Model+Attention'])

# Configurações do seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(14, 7))

# Criar gráfico de barras
ax = sns.barplot(
    data=df,
    x='common_size',
    y='BACC_mean',
    hue='Model+Attention',
    palette='tab20'
)

# Customizações
plt.title("Balanced Accuracy (BACC) para CNNs agrupadas por dimensão comum")
plt.ylabel("Balanced Accuracy (BACC)")
plt.xlabel("Common size of projected features")
plt.ylim(0, 1)
plt.legend(title="Modelo (Mecanismo de Atenção)", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig("bacc_all_common_sizes_grouped.png", bbox_inches='tight')
plt.show()