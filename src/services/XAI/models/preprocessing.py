import os
import pickle
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

# ==========================================================
# IMAGE
# ==========================================================

def load_image_transform(size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def process_image_pil(img_pil, device):
    transform = load_image_transform()
    tensor = transform(img_pil).unsqueeze(0).to(device)
    return tensor


# ==========================================================
# METADATA PAD-UFES-20
# ==========================================================

PAD_COLUMNS = [
    "patient_id","lesion_id","smoke","drink",
    "background_father","background_mother",
    "age","pesticide","gender",
    "skin_cancer_history","cancer_history",
    "has_piped_water","has_sewage_system",
    "fitspatrick","region",
    "diameter_1","diameter_2",
    "diagnostic",
    "itch","grew","hurt",
    "changed","bleed",
    "elevation","img_id","biopsed"
]

NUMERICAL_COLS = ["age","diameter_1","diameter_2"]
DROP_COLS = ["patient_id","lesion_id","img_id","biopsed","diagnostic"]


def clean_metadata(df):
    df = df.fillna("EMPTY")
    df = df.replace(r"^\s*$", "EMPTY", regex=True)
    df = df.replace("BRASIL", "BRAZIL")
    return df


def parse_csv_line_to_cols(text_line):
    parts = text_line.split(",")

    if len(parts) < len(PAD_COLUMNS):
        parts += [""] * (len(PAD_COLUMNS) - len(parts))
    elif len(parts) > len(PAD_COLUMNS):
        parts = parts[:len(PAD_COLUMNS)]

    return pd.DataFrame([parts], columns=PAD_COLUMNS)


def process_metadata_pad20(text_line, encoder_dir, device):

    df = parse_csv_line_to_cols(text_line)
    df = clean_metadata(df)

    features = df.drop(columns=DROP_COLS)

    categorical_cols = [c for c in features.columns
                        if c not in NUMERICAL_COLS]

    features[categorical_cols] = features[categorical_cols].astype(str)

    features[NUMERICAL_COLS] = (
        features[NUMERICAL_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )

    with open(os.path.join(encoder_dir,"ohe_pad_20.pickle"),"rb") as f:
        ohe = pickle.load(f)

    with open(os.path.join(encoder_dir,"scaler_pad_20.pickle"),"rb") as f:
        scaler = pickle.load(f)

    categorical_data = ohe.transform(features[categorical_cols])
    numerical_data = scaler.transform(features[NUMERICAL_COLS])

    processed = np.hstack([categorical_data, numerical_data])

    return torch.tensor(processed, dtype=torch.float32).to(device)