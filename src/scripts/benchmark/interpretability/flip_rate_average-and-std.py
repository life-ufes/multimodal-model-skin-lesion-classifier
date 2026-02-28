import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

from PIL import Image
from torchvision import transforms

# ==========================================================
# PATH SETUP
# ==========================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "./data/PAD-UFES-20"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
ENCODER_DIR = "./data/preprocess_data"

BASE_RESULTS_DIR = "./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/unfrozen_weights/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture"

OUT_DIR = "./results/flip_analysis_all_folds"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_LIST = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
K = len(CLASS_LIST)

FEATURE_LIST = ["age","region","gender","itch","grew","bleed",
                "changed","elevation","hurt"]

# ==========================================================
# IMAGE
# ==========================================================
def load_image_transform(size=(224,224)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

def process_image(path):
    img = Image.open(path).convert("RGB")
    tensor = load_image_transform()(img).unsqueeze(0).to(DEVICE)
    return tensor

# ==========================================================
# METADATA
# ==========================================================
df_all_meta = pd.read_csv(METADATA_PATH)

NUMERICAL_COLS = ["age","diameter_1","diameter_2"]
DROP_COLS = ["patient_id","lesion_id","img_id","biopsed","diagnostic"]

with open(os.path.join(ENCODER_DIR,"ohe_pad_20.pickle"),"rb") as f:
    OHE = pickle.load(f)

with open(os.path.join(ENCODER_DIR,"scaler_pad_20.pickle"),"rb") as f:
    SCALER = pickle.load(f)

def process_metadata_row(row_dict):
    df = pd.DataFrame([row_dict])
    features = df.drop(columns=DROP_COLS)

    categorical_cols = [c for c in features.columns if c not in NUMERICAL_COLS]
    features[categorical_cols] = features[categorical_cols].astype(str)
    features[NUMERICAL_COLS] = features[NUMERICAL_COLS].apply(
        pd.to_numeric, errors="coerce").fillna(-1)

    cat = OHE.transform(features[categorical_cols])
    num = SCALER.transform(features[NUMERICAL_COLS])
    processed = np.hstack([cat,num])
    return torch.tensor(processed,dtype=torch.float32).to(DEVICE)

# ==========================================================
# MODEL
# ==========================================================
from models import multimodalIntraInterModal

def load_model(model_path):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=K,
        device=DEVICE,
        cnn_model_name="densenet169",
        text_model_name="one-hot-encoder",
        vocab_size=91,
        num_heads=8,
        attention_mecanism="gfcam",
        n=2,
        unfreeze_weights="unfrozen_weights"
    )

    ckpt = torch.load(model_path, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model

# ==========================================================
# MUTATION
# ==========================================================
def mutate_metadata(row, feature):
    r = copy.deepcopy(row)

    if feature in ["itch","grew","bleed","changed","hurt","elevation",
                   "smoke","drink","skin_cancer_history","cancer_history"]:
        r[feature] = not bool(r[feature])

    elif feature in ["diameter_1","diameter_2"]:
        r[feature] = float(r[feature]) + 5
    
    elif feature=="age":
        r[feature] = 80

    elif feature == "gender":
        r[feature] = "MALE" if r[feature]=="FEMALE" else "FEMALE"

    elif feature == "region":
        r[feature] = "FACE" if r[feature]!="FACE" else "FOREARM"

    return r

# ==========================================================
# PREDICTION
# ==========================================================
def predict_class(model, image_tensor, metadata_tensor):
    with torch.no_grad():
        logits = model(image_tensor, metadata_tensor)
        probs = torch.softmax(logits, dim=1)
        return int(torch.argmax(probs, dim=1).item())

# ==========================================================
# FLIP PER FOLD
# ==========================================================
def run_fold(fold_number):

    print(f"\nRunning Fold {fold_number}")

    fold_dir = os.path.join(BASE_RESULTS_DIR, f"densenet169_fold_{fold_number}")
    model_path = os.path.join(fold_dir,"model.pth")
    pred_path = os.path.join(fold_dir,f"predictions_eval_fold_{fold_number}.csv")

    model = load_model(model_path)

    df_val = pd.read_csv(pred_path)
    used_images = df_val["image_name"].tolist()

    df_meta = df_all_meta[df_all_meta["img_id"].isin(used_images)]

    flip_counts = {f:0 for f in FEATURE_LIST}
    total = 0

    for _, row in df_meta.iterrows():

        image_path = os.path.join(IMAGE_DIR,row["img_id"])
        if not os.path.exists(image_path):
            continue

        image_tensor = process_image(image_path)
        meta_dict = row.to_dict()

        baseline_tensor = process_metadata_row(meta_dict)
        c0 = predict_class(model,image_tensor,baseline_tensor)

        for f in FEATURE_LIST:
            mutated = mutate_metadata(meta_dict,f)
            mutated_tensor = process_metadata_row(mutated)
            c1 = predict_class(model,image_tensor,mutated_tensor)

            if c1 != c0:
                flip_counts[f]+=1

        total+=1

    flip_rates = {f:flip_counts[f]/total for f in FEATURE_LIST}
    return flip_rates

# ==========================================================
# MAIN
# ==========================================================
def main():

    all_fold_results = []

    for fold in range(1,6):
        fold_result = run_fold(fold)
        all_fold_results.append(fold_result)

    df = pd.DataFrame(all_fold_results)
    df_mean = df.mean()
    df_std = df.std()

    summary = pd.DataFrame({
        "feature": df_mean.index,
        "mean_flip_rate": df_mean.values,
        "std_flip_rate": df_std.values
    }).sort_values("mean_flip_rate",ascending=False)

    summary.to_csv(os.path.join(OUT_DIR,"flip_summary_all_folds.csv"),index=False)

    plt.figure(figsize=(8,6))
    plt.barh(summary["feature"],summary["mean_flip_rate"])
    plt.xlabel("Mean Flip Rate (5 folds)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"flip_rate_all_folds.png"),dpi=400)
    plt.close()

    print("\nFINAL SUMMARY")
    print(summary)

if __name__=="__main__":
    main()