import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import ScoreCam
import gradcam 
from gradcam import GradCAM
import gradcam_plusplus
from gradcam_plusplus import GradCAMPlusPlus
from models import multimodalIntraInterModal

# ===== Main Script =====

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/unfreeze-weights/2/weighted-after-crossattention/model_densenet169_with_one-hot-encoder_512/densenet169_fold_5_20250112_181658/model.pth"
    
    # Carrega e pré-processa a imagem
    image_path = "./PAD-UFES-20/images/PAT_795_1508_925.png"
    image_pil = Image.open(image_path)
    processed_image = gradcam_plusplus.process_image(image_pil, image_encoder="densenet169")
    processed_image = processed_image.unsqueeze(0).to(device)  # Adiciona dimensão de batch

    # Processa a metadata
    column_names = [
        "patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age",
        "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
        "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic", "itch",
        "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"
    ]
    text = "PAT_795,1508,False,True,GERMANY,GERMANY,69,True,MALE,True,True,True,True,3.0,HAND,11.0,10.0,ACK,True,True,True,True,True,True,PAT_795_1508_925.png,True"
    metadata = gradcam_plusplus.process_data(text, column_names)
    processed_metadata = gradcam_plusplus.one_hot_encoding(metadata)
    processed_metadata_tensor = torch.tensor(processed_metadata, dtype=torch.float32).to(device)

    # Carrega o modelo multimodal
    model = gradcam_plusplus.load_multimodal_model(device, model_path, "weighted-after-crossattention")
    
    # Seleciona a camada alvo para os métodos de CAM
    target_layer = model.image_encoder.features[-1]  # Ajuste conforme necessário

    # Uso do ScoreCAM
    scorecam = ScoreCam.ScoreCAM(model, target_layer, device)

    # Generate heatmap using ScoreCAM
    heatmap = scorecam.generate_heatmap(processed_image, torch.tensor(processed_metadata, dtype=torch.float32).to(device), -1)

    # Remove hook after use
    scorecam.remove_hook()
    # Aplicar colormap ao heatmap
    heatmap_resized_scorecam = ScoreCam.resize_heatmap(heatmap, (image_pil.height, image_pil.width))

    # Instancia os métodos GradCAM e GradCAM++
    target_class = -1  # Ajuste o índice da classe alvo conforme necessário
    gradcam = GradCAM(model, target_layer, device)
    gradcam_pp = GradCAMPlusPlus(model, target_layer, device)

    # Gera os heatmaps
    heatmap_gradcam = gradcam.generate_heatmap(processed_image, processed_metadata_tensor, target_class)
    heatmap_gradcam_pp = gradcam_pp.generate_heatmap(processed_image, processed_metadata_tensor, target_class)

    # Redimensiona os heatmaps para o tamanho da imagem original (usando as dimensões da imagem PIL)
    heatmap_gradcam_resized = gradcam_plusplus.resize_heatmap(heatmap_gradcam, (image_pil.height, image_pil.width))
    heatmap_gradcam_pp_resized = gradcam_plusplus.resize_heatmap(heatmap_gradcam_pp, (image_pil.height, image_pil.width))

    # Plotagem dos resultados: Imagem original, GradCAM e GradCAM++
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    # Imagem original
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Imagem com GradCAM sobreposto
    axes[1].imshow(image_pil)
    axes[1].imshow(heatmap_resized_scorecam, cmap='jet', alpha=0.4)
    axes[1].set_title("Image with ScoreCAM")
    axes[1].axis('off')


    # Imagem com GradCAM sobreposto
    axes[2].imshow(image_pil)
    axes[2].imshow(heatmap_gradcam_resized, cmap='jet', alpha=0.4)
    axes[2].set_title("Image with GradCAM")
    axes[2].axis('off')

    # Imagem com GradCAM++ sobreposto
    axes[3].imshow(image_pil)
    axes[3].imshow(heatmap_gradcam_pp_resized, cmap='jet', alpha=0.4)
    axes[3].set_title("Image with GradCAM++")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()
