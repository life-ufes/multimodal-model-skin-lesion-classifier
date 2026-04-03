import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os

def plot_degradation_curves(results_directory="./src/results"):
    # 1. Configurar o estilo visual do gráfico
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 2. Buscar todos os arquivos de resumo gerados
    # Busca arquivos no padrão summary_*.csv
    search_pattern = os.path.join(results_directory, "summary_*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"Nenhum arquivo encontrado com o padrão {search_pattern}")
        return

    # 3. Ler e concatenar os dados de todos os experimentos
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_data.append(df)
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # 4. Plotar as curvas de cada mecanismo de atenção
    # Agrupamos por mecanismo para plotar uma linha por modelo
    mechanisms = full_df["mechanism"].unique()
    
    for mech in mechanisms:
        subset = full_df[full_df["mechanism"] == mech].sort_values("missing_rate")
        
        # X: Taxa de perda (0.0 a 0.7)
        x = subset["missing_rate"]
        # Y: Média do Balanced Accuracy
        y = subset["bacc_mean"]
        # Desvio Padrão
        std = subset["bacc_std"]
        
        # Plotando a linha principal (Média)
        plt.plot(x, y, marker='o', linewidth=2, label=f"{mech}")
        
        # Adicionando a área sombreada (Desvio Padrão)
        # O desvio padrão mostra a estabilidade entre os 5 folds
        plt.fill_between(x, y - std, y + std, alpha=0.15)

    # 5. Estilização refinada do gráfico
    plt.title("Resiliência do Modelo à Ausência de Metadados Clínicos", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Taxa de Omissão de Metadados (Missing Rate)", fontsize=12)
    plt.ylabel("Balanced Accuracy Médio (5-Fold)", fontsize=12)
    
    # Forçar os ticks do eixo X a baterem exatamente com os seus missing_rates
    plt.xticks(full_df["missing_rate"].unique())
    plt.ylim(0.3, 0.8) # Ajuste esses limites de acordo com a sua escala de acurácia
    
    # Posicionamento da legenda
    plt.legend(title="Mecanismo de Fusão", loc="lower left", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    
    # Salvar a imagem final
    output_img = os.path.join(results_directory, "BAcc_degradation_curve.png")
    plt.savefig(output_img, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Gráfico de degradação salvo em: {output_img}")

# Exemplo de chamada:
if __name__ == "__main__":
    plot_degradation_curves("./src/results")