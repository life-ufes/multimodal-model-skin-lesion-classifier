import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.preprocessing import process_image_pil, process_metadata_pad20
from models.model_loader import load_model, find_last_conv
from models.cam import GradCAMPlusPlus

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LIST = ["NEV","BCC","ACK","SEK","SCC","MEL"]

BASE_RESULTS_DIR = "./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/unfrozen_weights/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture"
ENCODER_DIR = "./data/preprocess_data"
fold_number=3
unfreeze_weights = "unfrozen_weights"
print("Loading model...")
fold_dir = os.path.join(BASE_RESULTS_DIR, f"densenet169_fold_{fold_number}")

MODEL = load_model(device="cuda", model_path=os.path.join(fold_dir, "model.pth"))
    

TARGET_LAYER = find_last_conv(MODEL.image_encoder)
CAM = GradCAMPlusPlus(MODEL, TARGET_LAYER)

print("Model ready.")


def run_inference(image_pil, metadata_text):

    image_tensor = process_image_pil(image_pil, DEVICE)

    metadata_tensor = process_metadata_pad20(
        metadata_text,
        ENCODER_DIR,
        DEVICE
    )

    with torch.no_grad():
        logits = MODEL(image_tensor, metadata_tensor)
        probs = torch.softmax(logits, dim=1)

    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()

    heatmap = CAM.generate(
        image_tensor,
        metadata_tensor,
        pred_class
    )

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image_pil)
    ax.imshow(heatmap, cmap="jet", alpha=0.4)
    ax.axis("off")

    title = f"{CLASS_LIST[pred_class]} | conf={confidence:.3f}"
    ax.set_title(title)

    fig.canvas.draw()
    result = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return result, title