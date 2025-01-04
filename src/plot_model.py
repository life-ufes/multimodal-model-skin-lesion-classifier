import torch
from torchviz import make_dot
import os
from models import multimodalIntraInterModal  # Adjust the import based on your project structure

# Initialize and load your model
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

# Dictionary to store module outputs
module_outputs = {}

# Define a forward hook function
def hook_fn(module, input, output):
    module_outputs[module] = output

# Register hooks on top-level modules
hooks = []
for name, module in model.named_children():
    hooks.append(module.register_forward_hook(hook_fn))

# Create dummy inputs
dummy_image = torch.randn(1, 3, 224, 224).to(device)
dummy_text = torch.randn(1, 172).to(device)

# Forward pass to trigger hooks
output = model(dummy_image, dummy_text)

# Remove hooks
for hook in hooks:
    hook.remove()

# Generate the computation graph based on module outputs
# We'll use the final output for visualization
dot = make_dot(output, params=dict(model.named_parameters()))

# Customize node labels to reflect module names
# Note: This requires mapping parameter names to module names
param_to_module = {}
for name, module in model.named_modules():
    for param_name, _ in module.named_parameters(recurse=False):
        full_param_name = f"{name}.{param_name}" if name else param_name
        param_to_module[full_param_name] = name if name else "root"

# Relabel nodes based on module names
for node in dot.body:
    if node.startswith('    '):
        node_content = node.strip()
        if '=' in node_content:
            var, expr = node_content.split('=', 1)
            var = var.strip()
            expr = expr.strip()
            # If expr is a parameter, map it to module
            if expr in param_to_module:
                module_name = param_to_module[expr]
                new_label = module_name
                # Replace the label
                new_node = f'    {var} [label="{new_label}"];'
                dot.body.remove(node)
                dot.body.append(new_node)

# Save the plot with a supported format
model_plot_path = "multimodal_model_architecture"
dot.format = "png"  # Choose 'png', 'pdf', etc.
dot.render(model_plot_path, cleanup=True)
print(f"Model architecture saved at {model_plot_path}.png")
