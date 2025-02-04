import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from models import multimodalIntraInterModal

# ===== Utility Functions (same as before) =====

def load_transforms(image_encoder="densenet169"):
    if image_encoder == "vit-base-patch16-224":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform

def process_image(img, image_encoder="densenet169"):
    image = img.convert("RGB")
    transform = load_transforms(image_encoder)
    return transform(image)

def load_multimodal_model(device, model_path, attention_mecanism, vocab_size=85):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=6,
        num_heads=2,
        device=device,
        cnn_model_name="densenet169",
        text_model_name="one-hot-encoder",
        vocab_size=vocab_size,
        attention_mecanism=attention_mecanism
    )
    model.to(device)
    model.eval()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model

def process_data(text, column_names):
    data = pd.DataFrame([text.split(',')], columns=column_names)
    data = data.fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").replace("NÃO  ENCONTRADO", "EMPTY")
    data = data.replace(r'^\s*$', 'EMPTY', regex=True)
    data = data.replace(['NÃO ENCONTRADO', 'n/a', None], 'EMPTY')
    data = data.replace("BRASIL", "BRAZIL")
    return data

def one_hot_encoding(metadata):
    dataset_features = metadata.drop(columns=['patient_id', 'lesion_id', 'img_id', 'biopsed', 'diagnostic'])
    for col in ['age', 'diameter_1', 'diameter_2', 'fitspatrick']:
        dataset_features[col] = pd.to_numeric(dataset_features[col], errors='coerce')
    categorical_cols = dataset_features.select_dtypes(include=['object', 'bool']).columns
    numerical_cols = dataset_features.select_dtypes(include=['float64', 'int64']).columns
    dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)
    dataset_features[numerical_cols] = dataset_features[numerical_cols].fillna(-1)
    print(f"{dataset_features[numerical_cols]}\n")
    dataset_features_categorical = dataset_features[categorical_cols]
    ohe_path = "./src/results/preprocess_data/ohe.pickle"
    if os.path.exists(ohe_path):
        with open(ohe_path, 'rb') as f:
            ohe = pickle.load(f)
        categorical_data = ohe.transform(dataset_features_categorical)
    else:
        raise FileNotFoundError("OneHotEncoder file not found.")
    scaler_path = "./src/results/preprocess_data/scaler.pickle"
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        numerical_data = scaler.transform(dataset_features[numerical_cols])
    else:
        raise FileNotFoundError("StandardScaler file not found.")
    processed_metadata = np.hstack((categorical_data, numerical_data))
    return processed_metadata

def resize_heatmap(heatmap, target_size):
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(heatmap_tensor, size=target_size, mode='bilinear', align_corners=False)
    return heatmap_resized.squeeze().numpy()

# ===== GradCAM++ Implementation =====

class GradCAMPlusPlus:
    def __init__(self, model, target_layer, device):
        """
        model: The loaded multimodal model.
        target_layer: The convolutional layer whose activations will be used.
        device: 'cuda' or 'cpu'
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device

    def forward_with_activation(self, image, metadata):
        """
        Performs a forward pass while capturing the target layer's activations.
        Returns:
            output: The model's output.
            activations: The raw activations from the target layer.
        """
        activations = None

        def hook(module, input, output):
            nonlocal activations
            activations = output  # Do not detach to allow gradient flow

        handle = self.target_layer.register_forward_hook(hook)
        output = self.model(image, metadata)
        handle.remove()

        if activations is None:
            raise RuntimeError("Could not capture activations. Check the target layer.")
        return output, activations

    def generate_heatmap(self, image, metadata, target_class):
        """
        Generates a GradCAM++ heatmap for the specified target class.
        image: Preprocessed image tensor (1, C, H, W).
        metadata: Processed metadata tensor.
        target_class: The index of the target class.
        Returns:
            Heatmap as a numpy array.
        """
        image.requires_grad = True  # Ensure gradients flow

        # Forward pass capturing activations
        output, activations = self.forward_with_activation(image, metadata)
        target_score = output[0, target_class]

        # Compute the first-order gradients w.r.t. the activations
        grads = torch.autograd.grad(target_score, activations, retain_graph=True, create_graph=True)[0]
        # Compute second and third order derivatives element-wise
        grads2 = grads ** 2
        grads3 = grads ** 3

        # Compute the alpha coefficients:
        # For each pixel (i,j) in each channel k:
        #   alpha_k^{ij} = grads2 / (2 * grads2 + activations * grads3)
        # We sum the term activations*grads3 over spatial dimensions (i,j)
        denominator = 2 * grads2 + torch.sum(activations * grads3, dim=(2, 3), keepdim=True) + 1e-7
        alpha = grads2 / denominator

        # Weights are computed as the sum over spatial dimensions of (alpha * ReLU(grads))
        relu_grads = F.relu(grads)
        weights = torch.sum(alpha * relu_grads, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)

        # Weighted combination of activations (over channels)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # Shape: (1, 1, fH, fW)
        cam = F.relu(cam)

        # Normalize the heatmap to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Upsample the heatmap to the input image size
        _, _, H, W = image.shape
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().detach().numpy()


# ===== Main Script for GradCAM++ =====

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/weighted-after-crossattention/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20250108_170320/model.pth"
    # model_path="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/weighted-after-crossattention/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20250108_170320/model.pth"
    model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/unfreeze-weights/2/weighted-after-crossattention/model_densenet169_with_one-hot-encoder_512/densenet169_fold_5_20250112_181658/model.pth"
    # Load and preprocess image
    image_path = "./PAD-UFES-20/images/PAT_795_1508_925.png"
    image_pil = Image.open(image_path)
    processed_image = process_image(image_pil, image_encoder="densenet169")
    processed_image = processed_image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Define column names for metadata processing
    column_names = [
        "patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age",
        "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
        "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic", "itch",
        "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"
    ]
    text="PAT_795,1508,False,True,GERMANY,GERMANY,69,True,MALE,True,True,True,True,3.0,HAND,11.0,10.0,ACK,False,False,False,False,False,False,PAT_795_1508_925.png,True"
    metadata = process_data(text, column_names)
    processed_metadata = one_hot_encoding(metadata)
    processed_metadata_tensor = torch.tensor(processed_metadata, dtype=torch.float32).to(device)
    
    # Load multimodal model
    model = load_multimodal_model(device, model_path, "weighted-after-crossattention")
    
    # Choose target layer for GradCAM++
    target_layer = model.image_encoder.features[-1]  # Adjust as needed
    
    gradcam_pp = GradCAMPlusPlus(model, target_layer, device)
    target_class = -1  # Adjust target class index as needed
    
    # Generate GradCAM++ heatmap
    heatmap = gradcam_pp.generate_heatmap(processed_image, processed_metadata_tensor, target_class)
    heatmap_resized = resize_heatmap(heatmap, (image_pil.height, image_pil.width))
    
    # Visualize the results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(image_pil)
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.4)
    axes[1].set_title("Image with GradCAM++")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
