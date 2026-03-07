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
import math

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

OUT_DIR = "./results/feature_entropy_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_LIST = ["NEV","BCC","ACK","SEK","SCC","MEL"]
K = len(CLASS_LIST)

FEATURE_LIST = [
"age","region","gender","itch","grew","bleed","changed","elevation",
"hurt","smoke","drink","pesticide","skin_cancer_history",
"cancer_history","has_piped_water","has_sewage_system"
]

BENIGN = ["NEV","ACK","SEK"]
MALIGNANT = ["BCC","SCC","MEL"]

benign_idx = [CLASS_LIST.index(c) for c in BENIGN]
malig_idx = [CLASS_LIST.index(c) for c in MALIGNANT]

# ==========================================================
# ENTROPY
# ==========================================================
def entropy(p):
    eps = 1e-12
    p = np.clip(p, eps, 1)
    return -np.sum(p*np.log(p))


# ==========================================================
# LOAD DATA
# ==========================================================
df_all_meta = pd.read_csv(METADATA_PATH)

with open(os.path.join(ENCODER_DIR,"ohe_pad_20.pickle"),"rb") as f:
    OHE = pickle.load(f)

with open(os.path.join(ENCODER_DIR,"scaler_pad_20.pickle"),"rb") as f:
    SCALER = pickle.load(f)

OHE_FEATURE_NAMES = list(OHE.feature_names_in_)
SCALER_FEATURE_NAMES = list(SCALER.feature_names_in_)


# ==========================================================
# IMAGE PROCESSING
# ==========================================================
def process_image(path):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    img = Image.open(path).convert("RGB")

    return transform(img).unsqueeze(0).to(DEVICE)


# ==========================================================
# METADATA PROCESSING
# ==========================================================
def process_metadata(row_dict):

    df = pd.DataFrame([row_dict])

    for c in OHE_FEATURE_NAMES:
        if c not in df.columns:
            df[c] = "unknown"

    cat_df = df[OHE_FEATURE_NAMES].astype(str)

    for c in SCALER_FEATURE_NAMES:
        if c not in df.columns:
            df[c] = -1

    num_df = df[SCALER_FEATURE_NAMES].apply(
        pd.to_numeric,errors="coerce"
    ).fillna(-1)

    cat = OHE.transform(cat_df)
    num = SCALER.transform(num_df)

    processed = np.hstack([cat,num])

    return torch.tensor(processed,dtype=torch.float32).to(DEVICE)


# ==========================================================
# LOAD MODEL
# ==========================================================
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

    ckpt = torch.load(model_path,map_location=DEVICE)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model.load_state_dict(state_dict,strict=False)

    model.to(DEVICE).eval()

    return model


# ==========================================================
# MUTATE FEATURE
# ==========================================================
def mutate_feature(row,feature):

    r = copy.deepcopy(row)

    if feature=="age":

        val = pd.to_numeric(r.get("age",50),errors="coerce")

        r["age"] = int(np.clip(val+20,0,100))

        return r

    if feature=="gender":

        g = str(r.get("gender","")).upper()

        r["gender"] = "MALE" if g=="FEMALE" else "FEMALE"

        return r

    if feature=="region":

        r["region"] = "FACE"

        return r

    val = str(r.get(feature,"")).lower()

    r[feature] = "False" if val in ["true","1"] else "True"

    return r


# ==========================================================
# PREDICTION
# ==========================================================
def predict(model,img,meta):

    with torch.no_grad():

        logits = model(img,meta)

        probs = torch.softmax(logits,dim=1)[0].cpu().numpy()

    pred = np.argmax(probs)

    return pred,probs


# ==========================================================
# PLOTS
# ==========================================================
def plot_transition_matrix(cm,feature,out_path):

    plt.figure(figsize=(6,5))

    plt.imshow(cm,cmap="Blues")

    plt.xticks(range(K),CLASS_LIST,rotation=45)
    plt.yticks(range(K),CLASS_LIST)

    for i in range(K):
        for j in range(K):

            plt.text(j,i,cm[i,j],ha="center",va="center")

    plt.title(f"Prediction Transition Matrix - {feature}")

    plt.xlabel("Mutated Prediction")
    plt.ylabel("Original Prediction")

    plt.tight_layout()

    plt.savefig(out_path,dpi=400)

    plt.close()


def plot_feature_sensitivity_map(flip_summary,entropy_summary,out_path):

    df = flip_summary.merge(entropy_summary,on="feature")

    x = df["mean_delta_entropy"]
    y = df["mean_flip_rate"]

    plt.figure(figsize=(8,6))

    plt.scatter(x,y)

    for _,row in df.iterrows():

        plt.text(
            row["mean_delta_entropy"],
            row["mean_flip_rate"],
            row["feature"]
        )

    plt.axhline(y.mean(),linestyle="--")
    plt.axvline(x.mean(),linestyle="--")

    plt.xlabel("Mean Δ Entropy")
    plt.ylabel("Flip Rate")

    plt.title("Feature Sensitivity Map")

    plt.tight_layout()

    plt.savefig(out_path,dpi=400)

    plt.close()


# ==========================================================
# RUN FOLD
# ==========================================================
def run_fold(fold):

    print("Running fold",fold)

    fold_dir = os.path.join(BASE_RESULTS_DIR,f"densenet169_fold_{fold}")

    model = load_model(os.path.join(fold_dir,"model.pth"))

    df_val = pd.read_csv(
        os.path.join(fold_dir,f"predictions_eval_fold_{fold}.csv")
    )

    df_meta = df_val.merge(
        df_all_meta,
        left_on="image_name",
        right_on="img_id"
    )

    flip = {f:0 for f in FEATURE_LIST}
    entropy_delta = {f:[] for f in FEATURE_LIST}
    transitions = {f:{"c0":[],"c1":[]} for f in FEATURE_LIST}

    total=0

    for _,row in df_meta.iterrows():

        img_path = os.path.join(IMAGE_DIR,row["img_id"])

        if not os.path.exists(img_path):
            continue

        img = process_image(img_path)

        meta_dict = row.to_dict()

        pred0,probs0 = predict(model,img,process_metadata(meta_dict))

        H0 = entropy(probs0)

        for f in FEATURE_LIST:

            mutated = mutate_feature(meta_dict,f)

            pred1,probs1 = predict(model,img,process_metadata(mutated))

            H1 = entropy(probs1)

            entropy_delta[f].append(H1-H0)

            transitions[f]["c0"].append(pred0)
            transitions[f]["c1"].append(pred1)

            if pred0!=pred1:
                flip[f]+=1

        total+=1

    flip = {f:flip[f]/total for f in FEATURE_LIST}

    entropy_mean = {
        f:np.mean(entropy_delta[f]) for f in FEATURE_LIST
    }

    return flip,entropy_mean,transitions


# ==========================================================
# MAIN
# ==========================================================
def main():

    flip_all=[]
    entropy_all=[]

    global_trans={f:{"c0":[],"c1":[]} for f in FEATURE_LIST}

    for fold in range(1,6):

        flip,entropy_mean,trans = run_fold(fold)

        flip_all.append(flip)
        entropy_all.append(entropy_mean)

        for f in FEATURE_LIST:

            global_trans[f]["c0"]+=trans[f]["c0"]
            global_trans[f]["c1"]+=trans[f]["c1"]

    df_flip=pd.DataFrame(flip_all)

    flip_summary=pd.DataFrame({
        "feature":df_flip.columns,
        "mean_flip_rate":df_flip.mean(),
        "std_flip_rate":df_flip.std()
    })

    flip_summary.to_csv(
        os.path.join(OUT_DIR,"flip_summary.csv"),
        index=False
    )

    df_entropy=pd.DataFrame(entropy_all)

    entropy_summary=pd.DataFrame({
        "feature":df_entropy.columns,
        "mean_delta_entropy":df_entropy.mean(),
        "std_delta_entropy":df_entropy.std()
    })

    entropy_summary.to_csv(
        os.path.join(OUT_DIR,"entropy_summary.csv"),
        index=False
    )

    plot_feature_sensitivity_map(
        flip_summary,
        entropy_summary,
        os.path.join(OUT_DIR,"feature_sensitivity_map.png")
    )

    matrix_dir=os.path.join(OUT_DIR,"transition_matrices")
    os.makedirs(matrix_dir,exist_ok=True)

    for f in FEATURE_LIST:

        c0=global_trans[f]["c0"]
        c1=global_trans[f]["c1"]

        cm=confusion_matrix(c0,c1,labels=range(K))

        pd.DataFrame(cm,index=CLASS_LIST,columns=CLASS_LIST).to_csv(
            os.path.join(matrix_dir,f"{f}_matrix.csv")
        )

        plot_transition_matrix(
            cm,
            f,
            os.path.join(matrix_dir,f"{f}_matrix.png")
        )

    print("Analysis finished")


if __name__=="__main__":

    main()