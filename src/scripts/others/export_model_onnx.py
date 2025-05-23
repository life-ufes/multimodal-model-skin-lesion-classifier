import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../benchmark')))
from benchmark.models import multimodalIntraInterModal # Adjust the import based on your project structure

# Initialize your model (ensure parameters match those used during training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = multimodalIntraInterModal.MultimodalModel(
    num_classes=6, 
    device=device, 
    cnn_model_name="densenet169", 
    text_model_name="one-hot-encoder", 
    vocab_size=172,
    attention_mecanism="crossattention"
)
model.to(device)
model.eval()

# Load your trained model weights
model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/weights/crossattention/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20241231_061130/model.pth"  # Update with your actual model path
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)

# Create dummy inputs matching the expected input dimensions
dummy_image = torch.randn(1, 3, 224, 224).to(device)
dummy_text = torch.randn(1, 172).to(device)

# Export the model to ONNX with opset_version=13
onnx_path = "multimodal_model.onnx"
try:
    torch.onnx.export(
        model, 
        (dummy_image, dummy_text), 
        onnx_path,
        export_params=True,
        opset_version=13,  # Updated opset version
        do_constant_folding=True,
        input_names=['image', 'text_metadata'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'text_metadata': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to {onnx_path} with opset version 13.")
except Exception as e:
    print(f"Error exporting the model: {e}")
