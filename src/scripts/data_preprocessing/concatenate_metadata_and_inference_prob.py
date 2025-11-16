import pandas as pd
import os

if __name__ == "__main__":
    INFERENCE_PROB = "./src/results/resultados-artigo-blocos-residuais/att-intramodal+residual+cross-attention-metadados/model_davit_tiny.msft_in1k_with_one-hot-encoder_512_with_best_architecture/davit_tiny.msft_in1k_fold_1/predictions_eval_fold_1.csv"
    METADATA_FILE = "./data/PAD-UFES-20/metadata.csv"

    # Carregar arquivos
    meta = pd.read_csv(METADATA_FILE)
    preds = pd.read_csv(INFERENCE_PROB)

    print("Colunas metadata:", meta.columns.tolist())
    print("Colunas predições:", preds.columns.tolist())

    # 🔥 Merge correto: img_id <-> image_name
    merged = pd.merge(
        meta,
        preds,
        left_on="img_id",
        right_on="image_name",
        how="inner",        # somente validação
        suffixes=("", "_pred")
    )

    print("Shape final:", merged.shape)
    print(merged.head())

    # Salvar arquivo final
    os.makedirs("./src/results", exist_ok=True)
    merged.to_csv("./src/results/pad_metadata_with_probs_fold_1.csv", index=False)

    print("\nArquivo salvo com sucesso!")
