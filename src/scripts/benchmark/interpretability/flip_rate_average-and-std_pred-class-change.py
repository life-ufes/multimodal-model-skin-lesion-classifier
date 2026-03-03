import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models import multimodalIntraInterModal

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "./data/PAD-UFES-20"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
ENCODER_DIR = "./data/preprocess_data"

BASE_RESULTS_DIR = "./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/unfrozen_weights/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture"

OUT_DIR = os.path.abspath("./results/flip_analysis_complete")
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving results to:", OUT_DIR)

CLASS_LIST = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
FEATURE_LIST = [
    "age","region","gender","itch","grew","bleed","changed","elevation",
    "hurt","smoke","drink","pesticide","skin_cancer_history",
    "cancer_history","has_piped_water","has_sewage_system"
]

NUMERICAL_COLS = ["age","diameter_1","diameter_2"]
DROP_COLS = ["patient_id","lesion_id","img_id","biopsed","diagnostic"]
K = len(CLASS_LIST)

BENIGN = ["NEV","ACK","SEK"]
MALIGNANT = ["BCC","SCC","MEL"]

benign_idx = [CLASS_LIST.index(c) for c in BENIGN]
malig_idx = [CLASS_LIST.index(c) for c in MALIGNANT]

# ==========================================================
# LOAD DATA
# ==========================================================
df_all_meta = pd.read_csv(METADATA_PATH)

with open(os.path.join(ENCODER_DIR,"ohe_pad_20.pickle"),"rb") as f:
    OHE = pickle.load(f)

with open(os.path.join(ENCODER_DIR,"scaler_pad_20.pickle"),"rb") as f:
    SCALER = pickle.load(f)

# ==========================================================
# PROCESSING FUNCTIONS
# ==========================================================
def load_image_transform(size=(224,224)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def process_image(path):
    img = Image.open(path).convert("RGB")
    return load_image_transform()(img).unsqueeze(0).to(DEVICE)

def process_metadata_row(row_dict):
    df = pd.DataFrame([row_dict])
    features = df.drop(columns=DROP_COLS, errors="ignore")

    categorical_cols = [c for c in features.columns if c not in NUMERICAL_COLS]
    features[categorical_cols] = features[categorical_cols].astype(str)
    features[NUMERICAL_COLS] = features[NUMERICAL_COLS]\
        .apply(pd.to_numeric, errors="coerce").fillna(-1)

    cat = OHE.transform(features[categorical_cols])
    num = SCALER.transform(features[NUMERICAL_COLS])
    processed = np.hstack([cat,num])
    return torch.tensor(processed,dtype=torch.float32).to(DEVICE)

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
    model.to(DEVICE).eval()
    return model

# ==========================================================
# MUTATION
# ==========================================================
def mutate_metadata(row, feature):
    r = copy.deepcopy(row)
    truthy = ["true","1","1.0","yes","y"]

    if feature in [
        "itch","grew","bleed","changed","hurt","elevation",
        "smoke","drink","pesticide","skin_cancer_history",
        "cancer_history","has_piped_water","has_sewage_system"
    ]:
        val = str(r.get(feature,"")).lower()
        r[feature] = "False" if val in truthy else "True"

    elif feature == "age":
        r[feature] = 80

    elif feature == "gender":
        r[feature] = "MALE" if str(r.get(feature,"")).upper()=="FEMALE" else "FEMALE"

    elif feature == "region":
        r[feature] = "FACE" if str(r.get(feature,"")).upper()!="FACE" else "FOREARM"

    return r

def predict_class(model, img_t, meta_t):
    with torch.no_grad():
        logits = model(img_t, meta_t)
        return int(torch.argmax(torch.softmax(logits,dim=1),dim=1).item())

# ==========================================================
# RUN FOLD
# ==========================================================
def run_fold(fold_number):
    print(f"\nRunning fold {fold_number}")
    fold_dir = os.path.join(BASE_RESULTS_DIR, f"densenet169_fold_{fold_number}")
    model = load_model(os.path.join(fold_dir,"model.pth"))

    df_val = pd.read_csv(os.path.join(fold_dir,f"predictions_eval_fold_{fold_number}.csv"))
    df_meta = df_all_meta[df_all_meta["img_id"].isin(df_val["image_name"])]

    fold_rates = {f:0 for f in FEATURE_LIST}
    fold_trans = {f:{"c0":[],"c1":[]} for f in FEATURE_LIST}
    y_true, y_pred = [], []
    total=0

    for _,row in df_meta.iterrows():
        img_path = os.path.join(IMAGE_DIR,row["img_id"])
        if not os.path.exists(img_path): continue

        img_t = process_image(img_path)
        meta_dict = row.to_dict()

        c0 = predict_class(model,img_t,process_metadata_row(meta_dict))

        diag = str(row["diagnostic"]).strip()
        if diag in CLASS_LIST:
            y_true.append(CLASS_LIST.index(diag))
            y_pred.append(c0)

        for f in FEATURE_LIST:
            mutated = mutate_metadata(meta_dict,f)
            c1 = predict_class(model,img_t,process_metadata_row(mutated))
            fold_trans[f]["c0"].append(c0)
            fold_trans[f]["c1"].append(c1)
            if c1!=c0:
                fold_rates[f]+=1

        total+=1

    fold_rates = {f:(fold_rates[f]/total if total>0 else 0) for f in FEATURE_LIST}
    return fold_rates, fold_trans, y_true, y_pred

# ==========================================================
# MAIN
# ==========================================================
def main():

    all_rates = []
    all_clinical_folds = []  # ← novo
    global_trans = {f: {"c0": [], "c1": []} for f in FEATURE_LIST}
    global_y_true, global_y_pred = [], []

    for fold in range(1, 6):

        rates, trans, y_t, y_p = run_fold(fold)
        all_rates.append(rates)

        # ===============================
        # Clinical transitions per fold
        # ===============================
        fold_clinical = []

        for f in FEATURE_LIST:

            tm = confusion_matrix(
                trans[f]["c0"],
                trans[f]["c1"],
                labels=range(K)
            )

            # Benign → Malignant
            b2m = tm[np.ix_(benign_idx, malig_idx)].sum()
            b_total = tm[benign_idx, :].sum()
            b2m_rate = b2m / (b_total + 1e-12)

            # Malignant → Benign
            m2b = tm[np.ix_(malig_idx, benign_idx)].sum()
            m_total = tm[malig_idx, :].sum()
            m2b_rate = m2b / (m_total + 1e-12)

            fold_clinical.append({
                "feature": f,
                "b2m_rate": b2m_rate,
                "m2b_rate": m2b_rate
            })

        all_clinical_folds.append(pd.DataFrame(fold_clinical))

        global_y_true.extend(y_t)
        global_y_pred.extend(y_p)

        for f in FEATURE_LIST:
            global_trans[f]["c0"].extend(trans[f]["c0"])
            global_trans[f]["c1"].extend(trans[f]["c1"])

    # ======================================================
    # 1️⃣ FLIP RATE SUMMARY
    # ======================================================
    df_rates = pd.DataFrame(all_rates)

    flip_summary = pd.DataFrame({
        "feature": df_rates.columns,
        "mean_flip_rate": df_rates.mean(),
        "std_flip_rate": df_rates.std()
    }).sort_values("mean_flip_rate", ascending=False)

    flip_summary.to_csv(os.path.join(OUT_DIR, "flip_summary.csv"), index=False)

    # ======================================================
    # 2️⃣ CLINICAL TRANSITION SUMMARY (MEAN ± STD)
    # ======================================================
    df_clin_all = pd.concat(all_clinical_folds)

    clin_summary = df_clin_all.groupby("feature").agg(
        mean_b2m=("b2m_rate", "mean"),
        std_b2m=("b2m_rate", "std"),
        mean_m2b=("m2b_rate", "mean"),
        std_m2b=("m2b_rate", "std")
    ).reset_index()

    clin_summary = clin_summary.sort_values("mean_b2m", ascending=False)

    clin_summary.to_csv(
        os.path.join(OUT_DIR, "clinical_transition_summary_mean_std.csv"),
        index=False
    )

    # ======================================================
    # 3️⃣ BARPLOT CLÍNICO COM ERRO
    # ======================================================
    plt.figure(figsize=(10, 6))

    y_pos = np.arange(len(clin_summary))

    plt.barh(
        y_pos - 0.2,
        clin_summary["mean_b2m"],
        xerr=clin_summary["std_b2m"],
        height=0.4,
        label="Benign → Malignant"
    )

    plt.barh(
        y_pos + 0.2,
        clin_summary["mean_m2b"],
        xerr=clin_summary["std_m2b"],
        height=0.4,
        label="Malignant → Benign"
    )

    plt.yticks(y_pos, clin_summary["feature"])
    plt.gca().invert_yaxis()
    plt.xlabel("Transition Rate (Mean ± STD across folds)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT_DIR, "clinical_transition_mean_std.png"),
        dpi=400
    )
    plt.close()

    print("\n✔ Clinical transition analysis (mean ± std) saved.")
    print(f"Results saved in: {OUT_DIR}")

if __name__=="__main__":
    try:
        print("Starting analysis...")
        main()
        print("Finished successfully.")
    except Exception as e:
        print("ERROR:", e)