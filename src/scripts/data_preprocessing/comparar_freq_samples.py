import numpy as np, glob, os
from sklearn.metrics import recall_score

CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]   # ordem do LabelEncoder

def resumo(pasta, titulo=""):
    # aceita a pasta do modelo (com vários folds) ou a de um fold só
    arqs_pred = sorted(glob.glob(os.path.join(pasta, "**", "predictions.npy"), recursive=True))
    if not arqs_pred:
        raise FileNotFoundError(f"Nenhum predictions.npy em {pasta}")
    arqs_lab = [f.replace("predictions.npy", "labels.npy") for f in arqs_pred]

    preds = np.concatenate([np.load(f) for f in arqs_pred])
    labs  = np.concatenate([np.load(f) for f in arqs_lab])

    print(f"\n=== {titulo} ({len(arqs_pred)} fold(s), {len(labs)} amostras)")
    print("classe:     ", "  ".join(f"{c:>5}" for c in CLASSES))
    print("verdadeiro: ", "  ".join(f"{n:>5}" for n in np.bincount(labs,  minlength=6)))
    print("previsto:   ", "  ".join(f"{n:>5}" for n in np.bincount(preds, minlength=6)))
    print("recall:     ", "  ".join(f"{r:>5.2f}" for r in recall_score(labs, preds, average=None, labels=range(6))))

base = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/artigo_1_GFCAM/07022026/PAD-UFES-20"
modelo = "model_densenet169_with_one-hot-encoder_512_with_best_architecture"

for metodo in ["gfcam", "concatenation"]:
    resumo(f"{base}/unfrozen_weights/8/{metodo}/{modelo}", f"{metodo} - fevereiro")
    resumo(f"{base}/multiclass/unfrozen_weights/8/{metodo}/{modelo}", f"{metodo} - hoje")