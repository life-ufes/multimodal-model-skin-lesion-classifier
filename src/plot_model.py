import torch
import hiddenlayer as hl
from models import multimodalIntraInterModal  # Adjust the import based on your project structure

# Initialize and load your model as before
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

# Create dummy inputs
dummy_image = torch.randn(1, 3, 224, 224).to(device)
dummy_text = torch.randn(1, 172).to(device)

# Generate hiddenlayer graph
graph = hl.build_graph(model, [dummy_image, dummy_text])
graph.theme = hl.graph.THEMES["blue"].copy()  # Optional: set theme
graph.save("multimodal_model_architecture", format="png")
print("Model architecture saved as 'multimodal_model_architecture.png'.")
