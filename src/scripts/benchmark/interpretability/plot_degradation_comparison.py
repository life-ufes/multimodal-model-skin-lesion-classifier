import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os


def plot_degradation_curves(results_directory="./src/results"):
    # Buscar todos os arquivos de resumo gerados
    search_pattern = os.path.join(results_directory, "summary_*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"Nenhum arquivo encontrado com o padrão {search_pattern}")
        return

    # Ler e concatenar os dados
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    # Garantir ordenação dos missing rates
    missing_rates_sorted = sorted(full_df["missing_rate"].unique())

    # Obter mecanismos e backbones únicos
    mechanisms = sorted(full_df["mechanism"].unique())
    backbones = sorted(full_df["backbone"].unique())

    for backbone in backbones:
        # Criar uma nova figura para cada backbone
        sns.set_theme(style="whitegrid", palette="muted")
        plt.figure(figsize=(10, 6), dpi=150)

        backbone_df = full_df[full_df["backbone"] == backbone]

        for mech in mechanisms:
            subset = backbone_df[backbone_df["mechanism"] == mech].sort_values("missing_rate")

            # Pular se não houver dados para esse mecanismo nesse backbone
            if subset.empty:
                continue

            x = subset["missing_rate"]
            y = subset["bacc_mean"]
            std = subset["bacc_std"]

            plt.plot(x, y, marker="o", linewidth=2, label=mech)
            plt.fill_between(x, y - std, y + std, alpha=0.15)

        plt.title(
            f"Resiliência do Modelo à Ausência de Metadados Clínicos\nBackbone: {backbone}",
            fontsize=14,
            fontweight="bold",
            pad=15
        )
        plt.xlabel("Taxa de Omissão de Metadados (Missing Rate)", fontsize=12)
        plt.ylabel("Balanced Accuracy Médio (5-Fold)", fontsize=12)

        plt.xticks(missing_rates_sorted)
        plt.ylim(0.3, 0.8)
        plt.legend(title="Mecanismo de Fusão", loc="lower left", fontsize=10, title_fontsize=11)
        plt.tight_layout()

        output_img = os.path.join(
            results_directory,
            f"BACC_degradation_curve_backbone_{backbone}.png"
        )
        plt.savefig(output_img, bbox_inches="tight")
        plt.close()

        print(f"📈 Gráfico de degradação salvo em: {output_img}")


if __name__ == "__main__":
    plot_degradation_curves("./src/results")