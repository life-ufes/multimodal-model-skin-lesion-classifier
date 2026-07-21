import os
import sys
import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

# ==========================================================
# PATH SETUP
# ==========================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models import multimodalIntraInterModal


# ==========================================================
# IMAGE TRANSFORM
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

def process_image(path, device):
    img = Image.open(path).convert("RGB")
    transform = load_image_transform()
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor


# ==========================================================
# LOAD MODEL (ROBUST CHECKPOINT LOADING)
# ==========================================================
def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) > 0 and keys[0].startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_multimodal_model(
    device,
    model_path,
    cnn_model_name="densenet169",
    attention_mecanism="gfcam",
    vocab_size=91,
    num_heads=8,
    n=2,
    num_classes=6,
    unfreeze_weights="frozen_weights"
):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=num_classes,
        device=device,
        cnn_model_name=cnn_model_name,
        text_model_name="one-hot-encoder",
        vocab_size=vocab_size,
        num_heads=num_heads,
        attention_mecanism=attention_mecanism,
        n=n,
        unfreeze_weights=unfreeze_weights
    )

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        state_dict = _strip_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = ckpt

    model.to(device)
    model.eval()
    return model


# ==========================================================
# METADATA PROCESSING (MATCH DATASET)
# ==========================================================
PAD_COLUMNS = [
    "patient_id", "lesion_id", "smoke", "drink",
    "background_father", "background_mother",
    "age", "pesticide", "gender",
    "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system",
    "fitspatrick", "region",
    "diameter_1", "diameter_2",
    "diagnostic",
    "itch", "grew", "hurt",
    "changed", "bleed",
    "elevation", "img_id", "biopsed"
]

NUMERICAL_COLS = ["age", "diameter_1", "diameter_2"]
DROP_COLS = ["patient_id", "lesion_id", "img_id", "biopsed", "diagnostic"]

def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("EMPTY")
    df = df.replace(r"^\s*$", "EMPTY", regex=True)
    df = df.replace(" ", "EMPTY").replace("  ", "EMPTY")
    df = df.replace("NÃO  ENCONTRADO", "EMPTY")
    df = df.replace("BRASIL", "BRAZIL")
    return df

def parse_csv_line_to_cols(text_line: str, columns: list) -> pd.DataFrame:
    parts = text_line.split(",")
    if len(parts) < len(columns):
        parts = parts + [""] * (len(columns) - len(parts))
    elif len(parts) > len(columns):
        parts = parts[:len(columns)]
    return pd.DataFrame([parts], columns=columns)

def process_metadata_pad20(text_line, encoder_dir, device):
    df = parse_csv_line_to_cols(text_line, PAD_COLUMNS)
    df = clean_metadata(df)

    features = df.drop(columns=DROP_COLS)
    categorical_cols = [c for c in features.columns if c not in NUMERICAL_COLS]

    features[categorical_cols] = features[categorical_cols].astype(str)
    features[NUMERICAL_COLS] = (
        features[NUMERICAL_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )

    ohe_path = os.path.join(encoder_dir, "ohe_pad_20.pickle")
    scaler_path = os.path.join(encoder_dir, "scaler_pad_20.pickle")

    if not os.path.exists(ohe_path):
        raise FileNotFoundError(f"OneHotEncoder not found: {ohe_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"StandardScaler not found: {scaler_path}")

    with open(ohe_path, "rb") as f:
        ohe = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    categorical_data = ohe.transform(features[categorical_cols])
    numerical_data = scaler.transform(features[NUMERICAL_COLS])
    processed = np.hstack([categorical_data, numerical_data])

    return torch.tensor(processed, dtype=torch.float32).to(device)


# ==========================================================
# FIND LAST CONV
# ==========================================================
def find_last_conv(module):
    last_conv = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found.")
    return last_conv


# ==========================================================
# BASE CAM CLASS
# ==========================================================
class BaseCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fh = self.target_layer.register_forward_hook(self._forward_hook)
        self._bh = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _normalize(self, cam):
        cam = F.relu(cam)
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def clear(self):
        self.activations = None
        self.gradients = None


# ==========================================================
# GradCAM
# ==========================================================
class GradCAM(BaseCAM):
    def generate(self, image, metadata, target_class):
        self.clear()
        image.requires_grad_(True)
        output = self.model(image, metadata)
        score = output[:, target_class]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations. Check target_layer.")
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = self._normalize(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()


# ==========================================================
# GradCAM++
# ==========================================================
class GradCAMPlusPlus(BaseCAM):
    def generate(self, image, metadata, target_class):
        self.clear()
        image.requires_grad_(True)
        output = self.model(image, metadata)
        score = output[:, target_class]
        if self.activations is None:
            raise RuntimeError("Activations not captured. Check forward hook / target_layer.")
        grads = torch.autograd.grad(
            score, self.activations, retain_graph=True, create_graph=True
        )[0]
        grads2 = grads ** 2
        grads3 = grads ** 3
        denominator = (
            2 * grads2 +
            torch.sum(self.activations * grads3, dim=(2, 3), keepdim=True) + 1e-8
        )
        alpha = grads2 / denominator
        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = self._normalize(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()


# ==========================================================
# UTILS
# ==========================================================
def sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
    return s

def resize_heatmap_to_image(heatmap, img_pil):
    return np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img_pil.size, Image.BILINEAR)
    ) / 255.0


def save_frozen_unfrozen_row(img_pil, res_frozen, res_unfrozen, out_path, class_list):
    """
    Single-row, 5-panel figure:
      [Original | Heatmap frozen | Overlay frozen | Heatmap unfrozen | Overlay unfrozen]
    """
    fig, ax = plt.subplots(1, 5, figsize=(26, 6))
    TITLE_SIZE = 24

    # Panel 0
    ax[0].imshow(img_pil)
    ax[0].set_title("A", fontsize=TITLE_SIZE, fontweight="bold")
    ax[0].axis("off")

    # Panel 1
    ax[1].imshow(res_frozen["heatmap"], cmap="jet")
    ax[1].set_title("B", fontsize=TITLE_SIZE, fontweight="bold")
    ax[1].axis("off")

    # Panel 2
    ax[2].imshow(img_pil)
    ax[2].imshow(res_frozen["heatmap"], cmap="jet", alpha=0.4)
    ax[2].set_title("C", fontsize=TITLE_SIZE, fontweight="bold")
    ax[2].axis("off")

    # Panel 3
    ax[3].imshow(res_unfrozen["heatmap"], cmap="jet")
    ax[3].set_title("D", fontsize=TITLE_SIZE, fontweight="bold")
    ax[3].axis("off")

    # Panel 4
    ax[4].imshow(img_pil)
    ax[4].imshow(res_unfrozen["heatmap"], cmap="jet", alpha=0.4)
    ax[4].set_title("E", fontsize=TITLE_SIZE, fontweight="bold")
    ax[4].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = "./data/PAD-UFES-20/images/PAT_46_881_14.png"
    encoder_dir = "./data/preprocess_data"
    class_list = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
    out_dir = "./results/XAI/24022026/"
    os.makedirs(out_dir, exist_ok=True)

    img_pil, image_tensor = process_image(image_path, device)

    text_configurations = {
        "original_metadata": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,True,True,True,PAT_46_881_14.png,True",
        "only_age":          "PAT_46,881,,,,,80,,,,,,,,,,,BCC,,,,,,,PAT_46_881_14.png,",
        "only_grew":         "PAT_46,881,,,,,,,,,,,,,,,,BCC,,True,,,,,PAT_46_881_14.png,",
        "only_bleed":        "PAT_46,881,,,,,,,,,,,,,,,,BCC,,,,,True,,PAT_46_881_14.png,",
        "only_changed":      "PAT_46,881,,,,,,,,,,,,,,,,BCC,,,,True,,,PAT_46_881_14.png,",
        "only_elevation":    "PAT_46,881,,,,,,,,,,,,,,,,BCC,,,,,,True,PAT_46_881_14.png,",
        "only_itch":         "PAT_46,881,,,,,,,,,,,,,,,,BCC,True,,,,,,PAT_46_881_14.png,",
        "only_hurt":         "PAT_46,881,,,,,,,,,,,,,,,,BCC,,,True,,,,PAT_46_881_14.png,",
        "only_region":       "PAT_46,881,,,,,,,,,,,,,NECK,,,BCC,,,,,,,PAT_46_881_14.png,"
    }

    for cam_type in ["gradcam", "gradcam++"]:

        # Collect results for both weight regimes
        results_by_status = {}

        for weight_status in ["frozen_weights", "unfrozen_weights"]:

            model_path = f"./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/{weight_status}/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_2/model.pth"

            model = load_multimodal_model(
                device=device,
                model_path=model_path,
                attention_mecanism="gfcam",
                cnn_model_name="densenet169",
                vocab_size=91,
                num_heads=8,
                n=2,
                num_classes=6,
                unfreeze_weights=weight_status
            )

            target_layer = find_last_conv(model.image_encoder)
            cam_generator = GradCAM(model, target_layer) if cam_type == "gradcam" \
                            else GradCAMPlusPlus(model, target_layer)

            results = {}
            for name, metadata_text in text_configurations.items():
                metadata_tensor = process_metadata_pad20(metadata_text, encoder_dir, device)

                with torch.no_grad():
                    logits = model(image_tensor, metadata_tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred_class].item()

                heatmap = cam_generator.generate(image_tensor, metadata_tensor, pred_class)
                heatmap = resize_heatmap_to_image(heatmap, img_pil)

                results[name] = {
                    "heatmap": heatmap,
                    "pred_class": pred_class,
                    "confidence": confidence
                }

            results_by_status[weight_status] = results

        # Build one frozen-vs-unfrozen figure per metadata configuration
        for name in text_configurations.keys():
            res_frozen = results_by_status["frozen_weights"][name]
            res_unfrozen = results_by_status["unfrozen_weights"][name]

            out_path = os.path.join(
                out_dir, f"{cam_type}_pad20_frozen_vs_unfrozen_{sanitize_filename(name)}.png"
            )
            save_frozen_unfrozen_row(
                img_pil=img_pil,
                res_frozen=res_frozen,
                res_unfrozen=res_unfrozen,
                out_path=out_path,
                class_list=class_list
            )
            print(f"Saved: {out_path}")